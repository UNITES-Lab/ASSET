# Datasets and Tasks

`uv run asset prepare <dataset>` downloads one dataset from Hugging Face, builds its splits, writes the LoRA training set to `data/llamafactory/`, and creates its tasks under `data/tasks/`.

| Dataset | Source | Tasks |
|---|---|---|
| `slake` | [BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) | `slake_<family>` |
| `radimgnet` | [raidium/RadImageNet-VQA](https://huggingface.co/datasets/raidium/RadImageNet-VQA) | `radimgnet_<family>` |
| `gmai` | [OpenGVLab/GMAI-MMBench](https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench) | `gmai_<family>` |
| `mimic` | [MiniMedMind/MIMIC-CXR-VQA](https://huggingface.co/datasets/MiniMedMind/MIMIC-CXR-VQA) | `mimic_pa_ap` |
| `vrb` | VQAv2 [train](https://huggingface.co/datasets/Multimodal-Fatima/VQAv2_train) and [validation](https://huggingface.co/datasets/Multimodal-Fatima/VQAv2_sample_validation) | `vrb_<corruption>` |

## Medical Acquisition-Quality Shifts

| Family | Modality | Severity parameter | Values |
|---|---|---|---|
| `ct_gaussian` | CT | image-noise std. | 60, 120, 180, 240 |
| `ct_poisson` | CT | dose fraction | 0.5, 0.25, 0.1, 0.05 |
| `mri_rician` | MRI | Rician noise σ | 0.05, 0.1, 0.15, 0.2, 0.25 |
| `mri_kspace` | MRI | acceleration factor R | 2, 4, 6, 8 |
| `vae` | CT and MRI | latent-noise σ | 1, 2, 3, 4, 5 |

The `vae` family is generated with `stabilityai/sd-vae-ft-mse`.

## Real Acquisition-Protocol Shift

`mimic_pa_ap` adapts a model trained on PA chest radiographs to AP chest radiographs and evaluates it on both views.

## Visual Robustness Benchmark

`vrb_<corruption>` covers 14 corruptions in 6 families, each at severities 1 to 5: noise (`shot_noise`, `gaussian_noise`, `impulse_noise`, `speckle_noise`), blur (`defocus_blur`, `zoom_blur`), weather (`snow`), image attributes (`brightness`, `contrast`, `saturate`), physical (`elastic_transform`, `spatter`), and digital (`pixelate`, `jpeg_compression`).

## Task Format

```text
data/tasks/<task>/
├── task.json
├── support.json
├── reference.json
├── eval/<condition>.json
└── images/<condition>/
```

`task.json` gives the modality, the shift family, the proxy bank, and the evaluation conditions. `support.json` holds the labeled target support examples, `reference.json` the no-shift reference examples, and `eval/` one evaluation set per condition.
