from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_task(
    task_dir: Path,
    modality: str,
    family: str,
    proxy_bank: str,
    support: list[dict],
    reference: list[dict],
    evaluation: dict[str, list[dict]],
) -> None:
    write_json(
        task_dir / "task.json",
        {"modality": modality, "family": family, "proxy_bank": proxy_bank, "conditions": list(evaluation)},
    )
    write_json(task_dir / "support.json", support)
    write_json(task_dir / "reference.json", reference)
    for condition, rows in evaluation.items():
        write_json(task_dir / "eval" / f"{condition}.json", rows)


def write_training_set(data_dir: Path, name: str, rows: list[dict]) -> None:
    root = data_dir / "llamafactory"
    write_json(root / f"{name}.json", rows)
    info_path = root / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info[name] = {
        "file_name": f"{name}.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    }
    write_json(info_path, info)


def relocate(row: dict, task_dir: Path, condition: str) -> dict:
    name = f"{row['image_id']}{Path(row['image_path']).suffix}"
    return dict(row, image_path=str(task_dir / "images" / condition / name))


def training_row(prompt: str, answer: str, image_path: str) -> dict:
    return {
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
        "images": [image_path],
    }
