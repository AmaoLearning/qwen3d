# Qwen-3D workspace reproduction

This directory makes the upstream repository portable. All generated assets stay
under the repository unless an environment variable explicitly overrides a path.

## Layout

- `../.venv`: Python 3.12 environment
- `../models/qwen3d`: Qwen-3D 3B and 7B checkpoints
- `../models/backbones`: matching Qwen2.5-VL 3B and 7B backbones
- `../data/refer_it_3d`: Nr3D, Sr3D, and ScanRefer annotations
- `../data/scannet_precomputed`: public UniVLG ScanNet metadata
- `../data/posed_rgbd`: licensed ScanNet/Matterport RGB-D assets
- `../output`: evaluation output

## Commands

```bash
source repro/activate.sh
python repro/verify_setup.py
bash repro/eval.sh 3b sr3d
bash repro/eval.sh 7b scanrefer_nr3d
```

默认已在复现脚本中关闭 WandB 上报：

```bash
USE_WANDB=False bash repro/eval.sh 3b sr3d
```

如需开启在线 WandB，可手动覆盖：

```bash
USE_WANDB=True WANDB_MODE=online bash repro/eval.sh 3b sr3d
```

`eval.sh` performs a preflight check before launching. The current host must have
an NVIDIA GPU and the licensed ScanNet RGB-D data. See `DATASETS.md` for the exact
status and manual acquisition steps.

The reproduction environment is offline by default and resolves the Qwen
backbone from `models/backbones`. The 7B launcher explicitly selects
`models/backbones/Qwen2.5-VL-7B-Instruct`; it will not contact Hugging Face.

ScanNet axis-alignment metadata is generated from the downloaded raw scenes by
`python repro/prepare_scannet_alignment.py` (the generated lookup contains
1513 scenes in this workspace).

The local CUDA 12.9 compiler is activated automatically when `.cuda` exists. To
build the remaining CUDA-only extension on a CUDA-capable host, run
`bash repro/build_pointops.sh`.

`install_env.sh` creates the Python 3.12 environment, installs the exact cu129
stack, and provisions a local CUDA 12.9 compiler. The CUDA compiler alone can
build portable A100/A6000/L40S objects; an NVIDIA driver/GPU is still required
to execute and validate them.

Network helpers retry at most three times and wait 120 seconds between attempts,
as requested.

Public downloads can be reproduced with:

```bash
python repro/download_public_data.py
python repro/download_qa_data.py
python repro/download_models.py
```
