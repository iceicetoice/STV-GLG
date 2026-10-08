from __future__ import annotations

import torch


def masked_sample_mse(prediction, target, mask):
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Each sample must have at least one valid common-future label.")
    return (((prediction - target) ** 2) * mask).sum(dim=1) / counts


def masked_mse(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Predictions, targets and validity masks must have identical shapes.")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("MSE requires a boolean mask with at least one valid position.")
    # Equals 1/(B*M) sum of squared errors when all labels are available.
    return (prediction[mask] - target[mask]).square().mean()


def progressive_objective(current: dict, previous: dict | None, batch: dict,
                          weight: float, gamma_normalized: float) -> tuple[torch.Tensor, dict]:
    area = masked_mse(current["prediction"], batch["target"], batch["target_mask"])
    progressive = area.new_zeros(())
    preceding = area.detach()
    if previous is not None:
        # Previous lead 1 predicts the newly observed day; leads 2.. align with current 1...
        aligned = previous["prediction"][:, 1:].detach()
        if aligned.shape != batch["target"].shape:
            raise ValueError("Misaligned adjacent-prefix future horizons.")
        preceding = masked_mse(aligned, batch["target"], batch["target_mask"])
        # Batch-mean hinge reproduces the manuscript's stated objective.
        progressive = torch.relu(area - preceding - gamma_normalized)
    total = area + weight * progressive
    return total, {"area": float(area.detach()), "previous": float(preceding.detach()),
                   "progressive": float(progressive.detach()), "total": float(total.detach())}
