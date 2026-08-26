"""Token-level 3D reweighted fine-tuning losses for Qwen-3D.

The paper defines a token weight from a blind (text-only) reference model and
applies it to the ordinary answer-token cross entropy.  This module keeps the
math independent from the Qwen-3D data/model plumbing so it can be unit-tested
without loading a language model.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


RFT_LOSS_TYPES = (
    "original",
    "paper_ratio",
    "neg_log_phi",
    "focal",
    "one_minus_log1p",
)


@dataclass(frozen=True)
class RFTLossOutput:
    loss: torch.Tensor
    statistics: Dict[str, torch.Tensor]


def _shared_answer_start(answer_start: torch.Tensor | int) -> int:
    """Return the common answer offset expected by Qwen-3D generation mode."""
    if isinstance(answer_start, int):
        return answer_start
    starts = answer_start.detach().reshape(-1)
    if starts.numel() == 0:
        raise ValueError("answer_start must contain at least one value")
    if not torch.all(starts == starts[0]):
        raise ValueError(
            "Qwen-3D's current generation head requires a shared answer_start "
            "within each batch"
        )
    return int(starts[0].item())


def answer_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    answer_start: torch.Tensor | int,
    *,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return answer-token NLL and its validity mask.

    Qwen-3D's modified generation head returns logits beginning one token
    before ``answer_start``.  Its final logit has no target, so dropping the
    final logit aligns the remaining logits with ``labels[:, answer_start:]``.
    """
    start = _shared_answer_start(answer_start)
    target = labels[:, start:]
    prediction = logits[:, :-1, :]
    if prediction.shape[:2] != target.shape:
        raise ValueError(
            "Qwen generation logits/labels are misaligned: "
            f"prediction={tuple(prediction.shape[:2])}, target={tuple(target.shape)}"
        )

    valid = target.ne(ignore_index)
    nll = F.cross_entropy(
        prediction.float().reshape(-1, prediction.shape[-1]),
        target.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(target)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    return nll, valid


def compute_rft_loss(
    *,
    full_logits: torch.Tensor,
    labels: torch.Tensor,
    answer_start: torch.Tensor | int,
    loss_type: str,
    phi_logits: Optional[torch.Tensor] = None,
    theta_blind_logits: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> RFTLossOutput:
    """Compute one mutually exclusive answer-token loss.

    ``phi_logits`` must come from the frozen base Qwen-VL with the LoRA adapter
    disabled and 3D inputs zeroed.  ``theta_blind_logits`` is additionally
    required by the paper-ratio variant and comes from the current LoRA model
    with the same blind input.
    """
    if loss_type not in RFT_LOSS_TYPES:
        raise ValueError(
            f"Unknown RFT loss type {loss_type!r}; expected one of {RFT_LOSS_TYPES}"
        )
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    full_nll, valid = answer_token_nll(full_logits, labels, answer_start)
    if not torch.any(valid):
        raise ValueError("RFT loss received no supervised answer tokens")
    valid_float = valid.to(full_nll.dtype)
    denominator = valid_float.sum().clamp_min(1.0)

    if loss_type == "original":
        weight = torch.ones_like(full_nll)
        p_phi = None
    else:
        if phi_logits is None:
            raise ValueError(f"phi_logits is required for loss type {loss_type!r}")
        phi_nll, phi_valid = answer_token_nll(phi_logits, labels, answer_start)
        if not torch.equal(valid, phi_valid):
            raise ValueError("full and phi validity masks differ")
        # exp(-NLL) is the base model probability assigned to the ground-truth
        # token.  Clamp only for numerical stability in subsequent transforms.
        p_phi = torch.exp(-phi_nll).clamp(min=eps, max=1.0)

        if loss_type == "paper_ratio":
            if theta_blind_logits is None:
                raise ValueError("theta_blind_logits is required for paper_ratio")
            theta_nll, theta_valid = answer_token_nll(
                theta_blind_logits, labels, answer_start
            )
            if not torch.equal(valid, theta_valid):
                raise ValueError("full and theta-blind validity masks differ")
            # Eq. (1): (-log p_phi) / (-log p_theta).  Both distributions are
            # evaluated without 3D input; the resulting weight is detached.
            weight = phi_nll / theta_nll.clamp_min(eps)
        elif loss_type == "neg_log_phi":
            weight = phi_nll
        elif loss_type == "focal":
            weight = (1.0 - p_phi).pow(gamma)
        elif loss_type == "one_minus_log1p":
            weight = 1.0 - torch.log1p(p_phi)
        else:  # pragma: no cover - guarded by RFT_LOSS_TYPES above
            raise AssertionError(loss_type)

    weight = torch.where(valid, weight.detach(), torch.zeros_like(weight))
    loss = (full_nll * weight).sum() / denominator

    valid_weight = weight[valid]
    stats = {
        "unweighted_loss": full_nll.sum().detach() / denominator,
        "weight_mean": valid_weight.mean().detach(),
        "weight_min": valid_weight.min().detach(),
        "weight_max": valid_weight.max().detach(),
        "valid_tokens": denominator.detach(),
    }
    if p_phi is not None:
        stats["p_phi_mean"] = p_phi[valid].mean().detach()
    return RFTLossOutput(loss=loss, statistics=stats)


def combine_post_training_losses(
    original_loss: torch.Tensor,
    rft_loss: Optional[torch.Tensor],
    *,
    original_coef: float,
    rft_coef: float,
) -> torch.Tensor:
    """Combine the preserved Qwen loss with one optional RFT loss."""
    if original_coef < 0 or rft_coef < 0:
        raise ValueError("post-training loss coefficients must be non-negative")
    if original_coef == 0 and (rft_loss is None or rft_coef == 0):
        raise ValueError("at least one active post-training loss coefficient is required")
    combined = original_coef * original_loss
    if rft_loss is not None:
        combined = combined + rft_coef * rft_loss
    return combined


__all__ = [
    "RFT_LOSS_TYPES",
    "RFTLossOutput",
    "answer_token_nll",
    "combine_post_training_losses",
    "compute_rft_loss",
]
