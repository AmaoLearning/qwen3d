# Qwen-3D LoRA post-training with Real-3DQA losses

`repro/train_rft_lora.py` adds token-level 3D reweighted fine-tuning to the
existing Qwen-3D generation loss. It currently accepts only genuine 3D QA
batches (`SQA3D` or `ScanQA`). The default selected variant is `focal`.

For an answer token `y_j`, let `p_phi` be the probability from the original,
frozen Qwen-VL backbone with all LoRA adapters disabled. The reference pass is
blind: Qwen-3D point-token embeddings and 3D RoPE coordinates are zeroed. The
current LoRA model's full 3D answer-token cross entropy is `CE_j`.

Available `--loss-type` values are:

- `original`: ordinary Qwen answer-token cross entropy only.
- `paper_ratio`: `w_j = (-log p_phi) / (-log p_theta_blind)`.
- `neg_log_phi`: `w_j = -log p_phi`.
- `focal`: `w_j = (1 - p_phi) ** gamma`.
- `one_minus_log1p`: `w_j = 1 - log(1 + p_phi)`.

Only one RFT variant can be active. For non-`original` variants, the language
loss before Qwen-3D's existing generation weight is

```text
L_language = original_loss_coef * mean(CE_j)
           + rft_loss_coef * mean(w_j * CE_j)
```

`MODEL.MASK_FORMER.GENERATION_WEIGHT` then balances this combined language loss
against any other Qwen-3D task losses. The launcher exposes all three controls
as `--original-loss-coef`, `--rft-loss-coef`, and `--generation-weight`.

## Safe one-step smoke run

```bash
cd /mnt/shared-storage-user/yicheng-data/Qwen-3D
source repro/activate.sh
python repro/train_rft_lora.py \
  --model-size 3b \
  --dataset sqa3d \
  --loss-type focal \
  --gamma 1 \
  --max-iter 1
```

Smoke mode is the default. It uses one 3D scene, performs the normal forward,
backward, and optimizer step, and suppresses evaluation and checkpoint writing.
Jobs longer than 10 iterations require the explicit `--full-run` flag.

Use `--dataset all` to concatenate the supported SQA3D and ScanQA training
sets and evaluate both validation sets. Full runs remove
`QWEN3D_SMOKE_SCENES` from the child environment even if it was exported by a
previous smoke job. Multi-GPU runs automatically enable Detectron2 unused
parameter discovery because Qwen-3D's conditional 3D heads do not all receive
gradients on every rank.

Full-run VQA evaluation uses pycocoevalcap METEOR and therefore requires a
Java runtime on the GPU node (Ubuntu: `apt-get install default-jre-headless`).

Single-node NCCL P2P is enabled by default so the 8xH200 NVSwitch fabric is
used for DDP collectives. Set `NCCL_P2P_DISABLE=1` explicitly only as a
compatibility fallback for a node without working CUDA peer access.

For example, the combined eight-GPU post-training entry point is:

```bash
python repro/train_rft_lora.py \
  --model-size 3b \
  --dataset all \
  --loss-type focal \
  --gamma 1 \
  --num-gpus 8 \
  --max-iter 13126 \
  --full-run
```

To exercise the requested focal exponents, run the command with `--gamma 0.5`,
`--gamma 1`, and `--gamma 2`. Outputs are isolated under
`output/rft_lora/<unique-run-name>`.
