from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from asset.model import load_model, lora_modules
from asset.scoring import candidate_scores, js_divergence, load_image, make_query, margin, masked_softmax, teacher_probs
from asset.subspace import gradient_bases, random_bases, register_subspace, remove_subspace
from asset.views import proxy_bank


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def influence_weights(reliability, sensitivity, smoothed, beta, gamma, uniform):
    reliability_std = float(torch.tensor(reliability, dtype=torch.float32).std(unbiased=False).item()) + 1e-6
    sensitivity_std = float(torch.tensor(sensitivity, dtype=torch.float32).std(unbiased=False).item()) + 1e-6
    if smoothed is None:
        smoothed = (list(reliability), list(sensitivity))
    else:
        smoothed = (
            [beta * old + (1.0 - beta) * new for old, new in zip(smoothed[0], reliability)],
            [beta * old + (1.0 - beta) * new for old, new in zip(smoothed[1], sensitivity)],
        )
    raw = []
    for c, s, c_bar, s_bar in zip(reliability, sensitivity, smoothed[0], smoothed[1]):
        if uniform:
            raw.append(1.0)
            continue
        c_term = torch.sigmoid(torch.tensor((c - c_bar) / reliability_std)).item()
        s_term = torch.sigmoid(torch.tensor((s - s_bar) / sensitivity_std)).item()
        raw.append(float(max(c_term * s_term, 1e-12) ** gamma))
    total = sum(raw) + 1e-12
    return [value / total for value in raw], smoothed


class Adaptation:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        task_dir = Path(args.task)
        task = read_json(task_dir / "task.json")
        support = read_json(task_dir / "support.json")[: args.support_size]
        reference = read_json(task_dir / "reference.json")[: args.reference_size]
        self.model, self.processor = load_model(args.adapter, args.device)
        self.modules = lora_modules(self.model)
        self.support = [(load_image(row["image_path"]), make_query(row)) for row in support]
        self.reference = [(load_image(row["image_path"]), make_query(row)) for row in reference]
        self.bank = proxy_bank(task, args.proxy_bank)
        images, queries = zip(*self.reference)
        self.teacher = teacher_probs(
            self.model, self.processor, args.device, list(images), list(queries), args.reference_batch_size
        )
        self.step = 0

    def score(self, image, query):
        return candidate_scores(self.model, self.processor, self.args.device, [image], [query])

    def write(self, log, record: dict) -> None:
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    def sensitivity(self, seed: int):
        views = self.args.sensitivity_views
        reliability = []
        sensitivity = []
        grads = {name: [] for name in self.modules}
        for index, (image, query) in enumerate(self.support):
            self.model.zero_grad(set_to_none=True)
            with torch.no_grad():
                anchor = self.score(image, query)[0][0]
            anchor_probs = anchor.softmax(dim=-1).detach().unsqueeze(0)
            mask = torch.ones_like(anchor_probs, dtype=torch.bool)
            value = 0.0
            for view in self.bank(image, views, seed + index):
                divergence = js_divergence(self.score(view, query)[0], anchor_probs, mask)
                (divergence / views).backward()
                value += float(divergence.detach().item()) / views
            for name, info in self.modules.items():
                a, b = info["a"].weight.grad, info["b"].weight.grad
                if a is None or b is None:
                    grads[name].append(torch.zeros(math.prod(info["a_shape"]) + math.prod(info["b_shape"])))
                else:
                    grads[name].append(torch.cat([a.flatten(), b.flatten()]).cpu())
            self.model.zero_grad(set_to_none=True)
            reliability.append(float(margin(anchor, query.target).item()))
            sensitivity.append(float(torch.tensor(value).item()))
        return reliability, sensitivity, grads

    def optimize(self, params, coefficients, round_index: int, log) -> None:
        args = self.args
        optimizer = torch.optim.AdamW(params, lr=args.lr_lora if coefficients is None else args.lr)
        rng = random.Random(args.seed + round_index * 1000 + 7)
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            support_indices = rng.sample(range(len(self.support)), k=min(args.support_batch_size, len(self.support)))
            reference_indices = rng.sample(
                range(len(self.reference)), k=min(args.reference_batch_size, len(self.reference))
            )
            losses = {"tgt": 0.0, "cons": 0.0, "src": 0.0, "pres": 0.0}
            count = len(support_indices) * args.consistency_views
            for local, index in enumerate(support_indices):
                image, query = self.support[index]
                scores, mask = self.score(image, query)
                target = F.cross_entropy(scores, torch.tensor([query.target], device=args.device))
                (target / len(support_indices)).backward()
                losses["tgt"] += float(target.detach().item()) / len(support_indices)
                probs = masked_softmax(scores, mask).detach()
                seed = args.seed + round_index * 100000 + self.step * 59 + local
                for view in self.bank(image, args.consistency_views, seed):
                    view_scores, view_mask = self.score(view, query)
                    consistency = js_divergence(view_scores, probs, mask & view_mask)
                    (args.lambda_cons * consistency / count).backward()
                    losses["cons"] += float(consistency.detach().item()) / count
            for index in reference_indices:
                image, query = self.reference[index]
                scores, mask = self.score(image, query)
                teacher = self.teacher[index].to(args.device).unsqueeze(0)
                source = F.cross_entropy(scores, torch.tensor([query.target], device=args.device))
                preserve = js_divergence(scores, teacher, mask)
                ((args.lambda_src * source + args.lambda_pres * preserve) / len(reference_indices)).backward()
                losses["src"] += float(source.detach().item()) / len(reference_indices)
                losses["pres"] += float(preserve.detach().item()) / len(reference_indices)
            total = losses["tgt"] + args.lambda_cons * losses["cons"]
            total += args.lambda_src * losses["src"] + args.lambda_pres * losses["pres"]
            if coefficients is not None:
                norm = torch.zeros((), device=args.device)
                for holder in coefficients:
                    norm = norm + holder.alpha.pow(2).sum()
                regularizer = args.lambda_alpha * norm
                regularizer.backward()
                total += float(regularizer.detach().item())
            optimizer.step()
            self.step += 1
            self.write(log, {"round": round_index, "step": self.step, "loss": total, **losses})

    def run(self) -> None:
        args = self.args
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        smoothed = None
        fixed = None
        with (output / "log.jsonl").open("w", encoding="utf-8") as log:
            for round_index in range(1, args.rounds + 1):
                if args.variant == "no_subspace":
                    params = [param for param in self.model.parameters() if param.requires_grad]
                    self.optimize(params, None, round_index, log)
                else:
                    if args.variant == "random_subspace":
                        bases = random_bases(self.modules, args.rank_cap, args.seed + round_index * 97)
                    elif args.variant == "fixed_subspace" and fixed is not None:
                        bases = fixed
                    else:
                        reliability, sensitivity, grads = self.sensitivity(args.seed + round_index * 100000)
                        weights, smoothed = influence_weights(
                            reliability, sensitivity, smoothed, args.beta, args.gamma, args.uniform_weighting
                        )
                        bases = gradient_bases(
                            self.modules, grads, weights, args.energy, args.rank_cap, args.energy_floor
                        )
                        if args.variant == "fixed_subspace" and bases:
                            fixed = bases
                    ranks = [basis.shape[1] for basis in bases.values()]
                    mean_rank = sum(ranks) / max(len(ranks), 1)
                    self.write(log, {"round": round_index, "modules": len(ranks), "mean_rank": mean_rank})
                    if bases:
                        coefficients = register_subspace(self.modules, bases)
                        self.optimize([holder.alpha for holder in coefficients], coefficients, round_index, log)
                        remove_subspace(self.modules, bases)
                self.model.save_pretrained(str(output / f"round_{round_index:02d}"))
                self.processor.save_pretrained(str(output / f"round_{round_index:02d}"))


def adapt(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Adaptation(args).run()
