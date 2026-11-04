from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(adapter: str | Path, device: str):
    config = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
    model = AutoModelForImageTextToText.from_pretrained(
        config["base_model_name_or_path"], dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
    processor = AutoProcessor.from_pretrained(str(adapter))
    return model.eval(), processor


def lora_modules(model) -> dict[str, dict]:
    modules = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and "default" in module.lora_A:
            a = module.lora_A["default"]
            b = module.lora_B["default"]
            modules[name] = {"a": a, "b": b, "a_shape": tuple(a.weight.shape), "b_shape": tuple(b.weight.shape)}
    return modules
