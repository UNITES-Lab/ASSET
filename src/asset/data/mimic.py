from __future__ import annotations

import json
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from asset.data import training_row, write_task, write_training_set

MIMIC = "MiniMedMind/MIMIC-CXR-VQA"
SEED = 20260724
SEX = re.compile(r"\b(gender|male|female|sex)\b", re.IGNORECASE)


def load(root: Path) -> tuple[dict[str, str], dict[str, list[dict]]]:
    archive = hf_hub_download(MIMIC, "images.zip", repo_type="dataset")
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(root)
    path = hf_hub_download(MIMIC, "full_test.json", repo_type="dataset")
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    view = {}
    for row in rows:
        answer = row["conversations"][1]["value"].strip().lower()
        if answer in {"pa", "ap"}:
            view[str(row["image"])] = answer
    clinical = defaultdict(list)
    for row in rows:
        image = str(row["image"])
        question = row["conversations"][0]["value"].replace("<image>", "").strip()
        answer = row["conversations"][1]["value"].strip().lower()
        if image not in view or answer not in {"yes", "no"} or SEX.search(question):
            continue
        clinical[image].append(
            {
                "image_id": Path(image).stem,
                "image_path": str((root / "images" / image).resolve()),
                "question": question,
                "answer": answer,
                "modality": "xray",
                "view": view[image].upper(),
            }
        )
    return view, clinical


def balanced(rows: list[dict], cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_answer = {answer: [row for row in rows if row["answer"] == answer] for answer in ("yes", "no")}
    for values in by_answer.values():
        rng.shuffle(values)
    count = min(cap // 2, *(len(values) for values in by_answer.values()))
    chosen = by_answer["yes"][:count] + by_answer["no"][:count]
    rng.shuffle(chosen)
    return chosen


def one_per_image(
    images: list[str], clinical: dict[str, list[dict]], count: int, seed: int
) -> tuple[list[dict], set[str]]:
    rng = random.Random(seed)
    remaining = list(images)
    rng.shuffle(remaining)
    desired = ["yes"] * (count // 2) + ["no"] * (count - count // 2)
    rng.shuffle(desired)
    rows = []
    chosen = []
    for answer in desired:
        index, image = next(
            (index, image)
            for index, image in enumerate(remaining)
            if any(row["answer"] == answer for row in clinical[image])
        )
        rows.append(rng.choice([row for row in clinical[image] if row["answer"] == answer]))
        chosen.append(image)
        remaining.pop(index)
    return rows, set(chosen)


def draw_support(pool: dict[str, list[dict]], seed: int) -> list[dict]:
    rng = random.Random(seed)
    available = list(pool)
    rng.shuffle(available)
    desired = ["yes"] * 16 + ["no"] * 16
    rng.shuffle(desired)
    chosen = []
    for answer in desired:
        image = rng.choice([image for image in available if any(row["answer"] == answer for row in pool[image])])
        available.remove(image)
        chosen.append(rng.choice([row for row in pool[image] if row["answer"] == answer]))
    rng.shuffle(chosen)
    return chosen


def rows_of(images: list[str], grouped: dict[str, list[dict]]) -> list[dict]:
    return [row for image in images for row in grouped[image]]


def prepare(data_dir: Path) -> None:
    view, clinical = load(data_dir / "mimic")
    pa = sorted(image for image, label in view.items() if label == "pa")
    ap = sorted(image for image, label in view.items() if label == "ap")
    rng = random.Random(SEED)
    rng.shuffle(pa)
    rng.shuffle(ap)

    reference, reference_images = one_per_image(pa, clinical, 32, SEED + 1)
    pa_rest = [image for image in pa if image not in reference_images]
    initial_support, initial_images = one_per_image(ap, clinical, 32, SEED + 2)
    ap_rest = [image for image in ap if image not in initial_images]
    train = balanced(rows_of(pa_rest[20:], clinical), 1000, SEED + 3)
    eval_pa = balanced(rows_of(pa_rest[:20], clinical), 300, SEED + 4)

    ap_rows = balanced(rows_of(ap_rest, clinical), 1000, SEED + 5)
    grouped = defaultdict(list)
    for row in ap_rows:
        grouped[row["image_path"]].append(row)
    ap_images = sorted(grouped)
    random.Random(SEED + 31).shuffle(ap_images)
    pool_rows = initial_support + balanced(rows_of(ap_images[:20], grouped), 200, SEED + 32)
    eval_ap = balanced(rows_of(ap_images[20:], grouped), 800, SEED + 33)
    pool = defaultdict(list)
    for row in pool_rows:
        pool[row["image_path"]].append(row)
    support = draw_support(pool, SEED + 100)

    write_training_set(
        data_dir,
        "mimic",
        [training_row(f"<image>{row['question']}", row["answer"], row["image_path"]) for row in train],
    )
    write_task(
        data_dir.resolve() / "tasks" / "mimic_pa_ap",
        "xray",
        "view",
        "medical",
        support,
        reference,
        {"ap": eval_ap, "pa": eval_pa},
    )
