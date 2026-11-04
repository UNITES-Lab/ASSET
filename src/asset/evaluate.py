from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from asset.model import load_model
from asset.scoring import candidate_scores, load_image, make_query


def accuracy(model, processor, device: str, rows: list[dict], batch_size: int) -> float:
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            queries = [make_query(row) for row in batch]
            images = [load_image(row["image_path"]) for row in batch]
            scores, _ = candidate_scores(model, processor, device, images, queries)
            predictions = scores.argmax(dim=-1).tolist()
            correct += sum(int(prediction == query.target) for prediction, query in zip(predictions, queries))
    return correct / len(rows)


def evaluate(args: argparse.Namespace) -> None:
    task_dir = Path(args.task)
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    model, processor = load_model(args.adapter, args.device)
    results = {}
    for condition in task["conditions"]:
        rows = json.loads((task_dir / "eval" / f"{condition}.json").read_text(encoding="utf-8"))
        results[condition] = accuracy(model, processor, args.device, rows, args.batch_size)
        print(json.dumps({"condition": condition, "accuracy": results[condition]}), flush=True)
    summary = {"task": task_dir.name, "adapter": str(args.adapter), "accuracy": results}
    if task["conditions"][0] == "clean":
        shifted = [results[condition] for condition in task["conditions"][1:]]
        summary["clean"] = results["clean"]
        summary["average"] = sum(shifted) / len(shifted)
        summary["strongest"] = shifted[-1]
    output = Path(args.output) if args.output else Path(args.adapter) / f"eval_{task_dir.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
