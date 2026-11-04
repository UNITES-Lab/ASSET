from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image

from asset.data import relocate, training_row, write_task, write_training_set
from asset.views import CORRUPTIONS, PROXY_SEVERITY, corrupt, stable_seed

TRAIN = "Multimodal-Fatima/VQAv2_train"
VALIDATION = "Multimodal-Fatima/VQAv2_sample_validation"
SEED = 42
SEVERITIES = [1, 2, 3, 4, 5]
PROMPT = "Answer with only yes or no."


def yes_no(rows, limit: int | None = None) -> list[dict]:
    chosen = []
    seen = set()
    for row in rows:
        answer = str(row.get("multiple_choice_answer") or "").strip().lower()
        if row["answer_type"] != "yes/no" or answer not in {"yes", "no"} or int(row["id_image"]) in seen:
            continue
        seen.add(int(row["id_image"]))
        chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


def train_rows():
    files = HfApi().list_repo_files(TRAIN, repo_type="dataset")
    for name in sorted(files):
        if name.startswith("data/train-") and name.endswith(".parquet"):
            yield from load_dataset(
                "parquet", data_files={"train": f"hf://datasets/{TRAIN}/{name}"}, split="train", streaming=True
            )


def record(row: dict, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    row["image"].convert("RGB").save(path, format="JPEG", quality=95)
    return {
        "image_id": int(row["id_image"]),
        "image_path": str(path.resolve()),
        "question": row["question"].strip(),
        "answer": str(row["multiple_choice_answer"]).strip().lower(),
    }


def corrupt_file(job: tuple) -> None:
    name, severity, subset, image_id, source, output = job
    array = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    corrupted = corrupt(array, name, severity, (SEED, subset, image_id, name, severity))
    Image.fromarray(corrupted).save(output, format="JPEG", quality=95)


def prepare(data_dir: Path, workers: int) -> None:
    root = data_dir.resolve() / "vrb"
    train = yes_no(train_rows(), 1000)
    validation = yes_no(load_dataset(VALIDATION, split="validation"))

    rng = random.Random(SEED)
    order = list(train)
    rng.shuffle(order)
    first = {int(row["id_image"]) for row in order[:16]}
    order = list(validation)
    rng.shuffle(order)
    support, evaluation = order[:16], order[16:316]
    rest = [row for row in train if int(row["id_image"]) not in first]
    random.Random(stable_seed(2026, "source_buffer_extension")).shuffle(rest)
    reference = [row for row in train if int(row["id_image"]) in first] + rest[:16]
    taken = {int(row["id_image"]) for row in support + evaluation}
    order = list(validation)
    random.Random(stable_seed(SEED, "target_val")).shuffle(order)
    support = support + [row for row in order if int(row["id_image"]) not in taken][:16]

    train_records = {
        int(row["id_image"]): record(row, root / "images" / "train" / f"{row['id_image']}.jpg") for row in train
    }
    reference = [train_records[int(row["id_image"])] for row in reference]
    support = [record(row, root / "images" / "validation" / f"{row['id_image']}.jpg") for row in support]
    evaluation = [record(row, root / "images" / "validation" / f"{row['id_image']}.jpg") for row in evaluation]
    write_training_set(
        data_dir,
        "vrb",
        [
            training_row(f"<image>\n{row['question']}\n{PROMPT}", row["answer"], row["image_path"])
            for row in train_records.values()
        ],
    )

    jobs = []
    for name, _ in CORRUPTIONS:
        task_dir = data_dir.resolve() / "tasks" / f"vrb_{name}"
        conditions = [f"{name}_{severity}" for severity in SEVERITIES]
        anchor = f"{name}_{PROXY_SEVERITY}"
        for severity, condition in zip(SEVERITIES, conditions):
            (task_dir / "images" / condition).mkdir(parents=True, exist_ok=True)
            for row in evaluation:
                output = relocate(row, task_dir, condition)["image_path"]
                jobs.append((name, severity, "target_eval", row["image_id"], row["image_path"], output))
        for row in support:
            output = relocate(row, task_dir, anchor)["image_path"]
            jobs.append((name, PROXY_SEVERITY, "target_support", row["image_id"], row["image_path"], output))
        write_task(
            task_dir,
            "natural",
            name,
            "corruption",
            [relocate(row, task_dir, anchor) for row in support],
            reference,
            {
                "clean": evaluation,
                **{condition: [relocate(row, task_dir, condition) for row in evaluation] for condition in conditions},
            },
        )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(corrupt_file, jobs, chunksize=16))
