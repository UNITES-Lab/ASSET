from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

from asset.data import training_row, write_json, write_training_set
from asset.data.shifts import build_tasks

SLAKE = "BoKelvin/SLAKE"
RADIMGNET = "raidium/RadImageNet-VQA"
GMAI = "OpenGVLab/GMAI-MMBench"
SEED = 2026
CLOSED = {"yes_no", "multiple_choice"}
MODALITIES = ["ct", "mri"]
TRAIN_PROMPT = "Answer the question concisely with a single word or short phrase."


def normalize(text) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'`")
    text = re.sub(r"[.,;:!?]+$", "", text)
    return text.strip()


def thumbnail_hash(image: Image.Image) -> str:
    with image.convert("RGB") as rgb:
        return hashlib.md5(rgb.resize((64, 64), Image.BICUBIC).tobytes()).hexdigest()


def save_image(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        image.convert("RGB").save(path)
    return str(path.resolve())


def clean_choice(choice) -> str:
    return re.sub(r"^[A-E]\s*[:.)]\s*", "", str(choice or "").strip(), flags=re.IGNORECASE).strip()


def answer_letter(answer, choices: list[str]) -> str:
    text = normalize(answer)
    if len(text) == 1 and text.isalpha():
        return text.upper()
    for index, choice in enumerate(choices):
        if normalize(choice) == text:
            return chr(65 + index)
    return str(answer or "").strip()


def make_record(
    image_id: str, image_path: str, question_id, question, answer, choices: list[str], modality: str
) -> dict:
    choices = [choice for choice in choices if str(choice).strip()]
    if choices:
        kind = "multiple_choice"
    elif normalize(answer) in {"yes", "no"}:
        kind = "yes_no"
    else:
        kind = "other"
    return {
        "image_id": image_id,
        "image_path": image_path,
        "question_id": str(question_id),
        "question": str(question or "").strip(),
        "answer": answer_letter(answer, choices) if choices else str(answer or "").strip(),
        "choices": choices,
        "answer_type": kind,
        "modality": modality,
    }


def load_slake(root: Path) -> dict[str, list[dict]]:
    archive = hf_hub_download(SLAKE, "imgs.zip", repo_type="dataset")
    images = root / "images"
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            name = Path(member.filename)
            if member.is_dir() or name.parts[0] != "imgs":
                continue
            target = images.joinpath(*name.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(handle.read(member))
    records = defaultdict(list)
    for split in ["train", "validation", "test"]:
        path = hf_hub_download(SLAKE, f"{split}.json", repo_type="dataset")
        for index, row in enumerate(json.loads(Path(path).read_text(encoding="utf-8"))):
            modality = str(row.get("modality", "")).lower()
            if row.get("q_lang") != "en" or modality not in {"ct", "mri", "x-ray", "xray"}:
                continue
            records[split].append(
                make_record(
                    f"slake_{row.get('img_id')}",
                    str((images / str(row["img_name"])).resolve()),
                    row.get("qid", index),
                    row.get("question", ""),
                    row.get("answer", ""),
                    [],
                    "xray" if modality in {"x-ray", "xray"} else modality,
                )
            )
    return records


def load_radimgnet(root: Path) -> list[dict]:
    records = []
    rows = load_dataset(RADIMGNET, name="benchmark", split="test", streaming=True)
    for index, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        modality = str(metadata.get("modality", "")).lower()
        if modality not in {"ct", "mri"}:
            continue
        image = row["image"].convert("RGB")
        image_id = f"radimgnet_benchmark_{thumbnail_hash(image)}"
        records.append(
            make_record(
                image_id,
                save_image(image, root / "images" / f"{image_id}.png"),
                metadata.get("question_id", index),
                row.get("question", ""),
                row.get("answer", ""),
                list(row.get("choices") or []),
                modality,
            )
        )
    return records


def load_gmai(root: Path) -> list[dict]:
    records = []
    path = hf_hub_download(GMAI, "GMAI_mm_bench_VAL.tsv", repo_type="dataset")
    table = pd.read_csv(path, sep="\t")
    for index, row in table.iterrows():
        modality = str(row.get("modality", "")).lower()
        if "ct" in modality and "mri" not in modality:
            modality = "ct"
        elif "mri" in modality:
            modality = "mri"
        else:
            continue
        choices = [clean_choice(row.get(letter)) for letter in ["A", "B", "C", "D", "E"]]
        choices = [choice for choice in choices if choice and choice.lower() != "nan"]
        answer = str(row.get("answer", "")).strip()
        if not answer:
            category = normalize(row.get("category", ""))
            for position, choice in enumerate(choices):
                if normalize(choice) == category:
                    answer = chr(65 + position)
                    break
        image = Image.open(io.BytesIO(base64.b64decode(str(row["image"])))).convert("RGB")
        image_id = f"gmai_val_{thumbnail_hash(image)}"
        records.append(
            make_record(
                image_id,
                save_image(image, root / "images" / f"{image_id}.jpg"),
                row.get("index", index),
                row.get("question", ""),
                answer,
                choices,
                modality,
            )
        )
    return records


def row_key(row: dict) -> tuple[str, str]:
    return row["image_id"], row["question_id"]


def filter_rows(rows, closed_only=False, modalities=None, exclude_images=None, exclude_rows=None) -> list[dict]:
    exclude_images = exclude_images or set()
    exclude_rows = exclude_rows or set()
    return [
        row
        for row in rows
        if row["image_id"] not in exclude_images
        and row_key(row) not in exclude_rows
        and (modalities is None or row["modality"] in modalities)
        and (not closed_only or row["answer_type"] in CLOSED)
    ]


def row_priority(row: dict) -> tuple[int, int, int, str]:
    return (
        0 if row["answer_type"] in CLOSED else 1,
        0 if row["answer_type"] == "multiple_choice" else 1,
        0,
        str(row["question_id"]),
    )


def unique_by_image(
    rows, limit, seed, closed_only=False, modalities=None, exclude_images=None, exclude_rows=None
) -> list[dict]:
    groups = defaultdict(list)
    for row in filter_rows(rows, closed_only, modalities, exclude_images, exclude_rows):
        groups[row["image_id"]].append(row)
    chosen = [dict(sorted(group, key=row_priority)[0]) for _, group in sorted(groups.items())]
    random.Random(seed).shuffle(chosen)
    return chosen[:limit]


def any_rows(
    rows, limit, seed, closed_only=False, modalities=None, exclude_images=None, exclude_rows=None
) -> list[dict]:
    rng = random.Random(seed)
    candidates = [dict(row) for row in filter_rows(rows, closed_only, modalities, exclude_images, exclude_rows)]
    rng.shuffle(candidates)
    candidates.sort(key=lambda row: (row["modality"], row_priority(row)))
    rng.shuffle(candidates)
    return candidates[:limit]


def image_ids(rows: list[dict]) -> set[str]:
    return {row["image_id"] for row in rows}


def select_balanced(rows, total, seed, closed_only, exclude_images=None, multi_question=False) -> list[dict]:
    exclude_images = exclude_images or set()
    targets = {modality: total // len(MODALITIES) for modality in MODALITIES}
    for modality in MODALITIES[: total % len(MODALITIES)]:
        targets[modality] += 1
    parts = {}
    keys = set()
    used = set()
    for offset, modality in enumerate(MODALITIES):
        parts[modality] = unique_by_image(
            rows, targets[modality], seed + offset, closed_only, {modality}, exclude_images | used
        )
        keys |= {row_key(row) for row in parts[modality]}
        used |= image_ids(parts[modality])
    if multi_question:
        for offset, modality in enumerate(MODALITIES):
            missing = targets[modality] - len(parts[modality])
            if missing <= 0:
                continue
            fill = any_rows(rows, missing, seed + 31 + offset, closed_only, {modality}, exclude_images, keys)
            parts[modality].extend(fill)
            keys |= {row_key(row) for row in fill}
            used |= image_ids(fill)
    selected = [
        part[index]
        for index in range(max(len(part) for part in parts.values()))
        for part in parts.values()
        if index < len(part)
    ]
    if len(selected) < total:
        fill = unique_by_image(
            rows, total - len(selected), seed + 101, closed_only, set(MODALITIES), exclude_images | used, keys
        )
        selected.extend(fill)
        keys |= {row_key(row) for row in fill}
    if len(selected) < total and multi_question:
        selected.extend(
            any_rows(rows, total - len(selected), seed + 211, closed_only, set(MODALITIES), exclude_images, keys)
        )
    return selected[:total]


def split_slake(records: dict[str, list[dict]]) -> dict[str, list[dict]]:
    reference_pool = select_balanced(records["train"], 64, SEED + 21, True)
    return {
        "train": select_balanced(records["train"], 1000, SEED + 22, False, image_ids(reference_pool), True),
        "reference_pool": reference_pool,
        "support_pool": select_balanced(records["validation"], 64, SEED + 23, True),
        "eval": select_balanced(records["test"], 300, SEED + 24, True, multi_question=True),
    }


def split_internal(records: list[dict]) -> dict[str, list[dict]]:
    support_pool = select_balanced(records, 64, SEED + 11, True)
    used = image_ids(support_pool)
    evaluation = select_balanced(records, 300, SEED + 12, True, used, True)
    used |= image_ids(evaluation)
    reference_pool = select_balanced(records, 64, SEED + 13, True, used)
    used |= image_ids(reference_pool)
    return {
        "train": select_balanced(records, 1000, SEED + 14, False, used, True),
        "reference_pool": reference_pool,
        "support_pool": support_pool,
        "eval": evaluation,
    }


def training_prompt(row: dict) -> str:
    question = row["question"]
    if row["choices"]:
        question = f"{question}\nChoices: " + " ".join(
            f"{chr(65 + index)}) {choice}" for index, choice in enumerate(row["choices"])
        )
    return f"<image>\n{question}\n{TRAIN_PROMPT}"


def prepare(dataset: str, data_dir: Path, workers: int, device: str) -> None:
    root = data_dir / dataset
    if dataset == "slake":
        splits = split_slake(load_slake(root))
    elif dataset == "radimgnet":
        splits = split_internal(load_radimgnet(root))
    else:
        splits = split_internal(load_gmai(root))
    for name, rows in splits.items():
        write_json(root / f"{name}.json", rows)
    write_training_set(
        data_dir,
        dataset,
        [training_row(training_prompt(row), row["answer"], row["image_path"]) for row in splits["train"]],
    )
    build_tasks(dataset, data_dir, splits, workers, device)
