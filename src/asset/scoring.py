from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class Query:
    prompt: str
    candidates: list[str]
    target: int


def normalize(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'`")
    text = re.sub(r"[.,;:!?]+$", "", text)
    return text.strip()


def choice_letter(answer: str, choices: Sequence[str]) -> str | None:
    answer = normalize(answer)
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()
    for index, choice in enumerate(choices):
        if normalize(choice) == answer:
            return chr(65 + index)
    return None


def build_prompt(question: str, choices: Sequence[str]) -> str:
    if choices:
        options = " ".join(f"{chr(65 + index)}) {choice}" for index, choice in enumerate(choices))
        return f"{question}\nOptions: {options}\nAnswer with only the letter of the correct option."
    if question.strip().lower().startswith(("is ", "are ", "does ", "do ", "was ", "were ")):
        return f"{question}\nAnswer with only yes or no."
    return f"{question}\nAnswer the question concisely with a single word or short phrase."


def make_query(row: dict) -> Query:
    choices = row.get("choices") or []
    if choices:
        candidates = [chr(65 + index) for index in range(len(choices))]
        target = candidates.index(choice_letter(row["answer"], choices))
    else:
        candidates = ["yes", "no"]
        target = candidates.index(str(row["answer"]).strip().lower())
    return Query(build_prompt(row["question"], choices), candidates, target)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def chat_text(processor, image: Image.Image, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def candidate_suffix(text: str, candidate: str) -> str:
    return candidate if text.endswith((" ", "\n")) else f" {candidate}"


def single_token_ids(text: str, candidates: Sequence[str], tokenizer) -> list[int] | None:
    ids = []
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate_suffix(text, candidate), add_special_tokens=False)
        if len(token_ids) != 1:
            return None
        ids.append(int(token_ids[0]))
    return ids


def pad(rows: Sequence[torch.Tensor], value: float) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(int(row.numel()) for row in rows)
    padded = torch.full((len(rows), width), value, dtype=rows[0].dtype, device=rows[0].device)
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=rows[0].device)
    for index, row in enumerate(rows):
        padded[index, : row.numel()] = row
        mask[index, : row.numel()] = True
    return padded, mask


def processor_images(processor, images: Sequence[Image.Image]) -> list:
    if "gemma3" in processor.__class__.__name__.lower():
        return [[image] for image in images]
    return list(images)


def sequence_scores(
    model, processor, device: str, image: Image.Image, text: str, candidates: Sequence[str]
) -> torch.Tensor:
    prompt = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    prompt_length = int(prompt["attention_mask"].sum(dim=1)[0].item())
    scores = []
    for candidate in candidates:
        inputs = processor(
            text=[text + candidate_suffix(text, candidate)], images=[image], return_tensors="pt", padding=True
        ).to(device)
        length = int(inputs["attention_mask"].sum(dim=1)[0].item())
        answer = inputs["input_ids"][0, prompt_length:length]
        logits = model(**inputs, use_cache=False).logits[0, prompt_length - 1 : length - 1]
        scores.append(F.log_softmax(logits.float(), dim=-1).gather(1, answer.unsqueeze(1)).squeeze(1).sum())
    return torch.stack(scores)


def candidate_scores(
    model, processor, device: str, images: Sequence[Image.Image], queries: Sequence[Query]
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = [chat_text(processor, image, query.prompt) for image, query in zip(images, queries)]
    ids = [single_token_ids(text, query.candidates, processor.tokenizer) for text, query in zip(texts, queries)]
    if any(row is None for row in ids):
        rows = [
            sequence_scores(model, processor, device, image, text, query.candidates)
            for image, text, query in zip(images, texts, queries)
        ]
        return pad(rows, float("-inf"))
    batch = processor_images(processor, images)
    inputs = processor(text=texts, images=batch, return_tensors="pt", padding=True).to(device)
    lengths = inputs["attention_mask"].sum(dim=1).long()
    logits = model(**inputs, use_cache=False).logits[torch.arange(len(images), device=device), lengths - 1].float()
    index, mask = pad([torch.tensor(row, dtype=torch.long, device=device) for row in ids], 0)
    return logits.gather(1, index).masked_fill(~mask, float("-inf")), mask


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    return torch.where(mask, probs, torch.zeros_like(probs))


def js_divergence(scores: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = masked_softmax(scores.float(), mask)
    target = torch.where(mask, target, torch.zeros_like(target))
    mean = torch.where(mask, 0.5 * (probs + target), torch.zeros_like(probs))
    mean = (mean / mean.sum(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12)
    target_term = torch.where(mask, target * (target.clamp_min(1e-12).log() - mean.log()), torch.zeros_like(target))
    probs_term = torch.where(mask, probs * (probs.clamp_min(1e-12).log() - mean.log()), torch.zeros_like(probs))
    return 0.5 * (target_term.sum() + probs_term.sum()) / max(int(scores.shape[0]), 1)


def margin(scores: torch.Tensor, target: int) -> torch.Tensor:
    if scores.numel() <= 1:
        return scores[target]
    others = torch.ones_like(scores, dtype=torch.bool)
    others[target] = False
    return scores[target] - torch.max(scores[others])


def teacher_probs(
    model, processor, device: str, images: Sequence[Image.Image], queries: Sequence[Query], batch_size: int
) -> list[torch.Tensor]:
    teacher = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = queries[start : start + batch_size]
            scores, _ = candidate_scores(model, processor, device, images[start : start + batch_size], batch)
            for row, query in zip(scores.softmax(dim=-1), batch):
                teacher.append(row[: len(query.candidates)].detach().cpu())
    return teacher
