import torch
import torch.nn.functional as F


def trajectory_loss(
    predictions: dict[str, torch.Tensor],
    gt_deltas: torch.Tensor,
    gt_positions: torch.Tensor,
    future_mask: torch.Tensor,
    delta_weight: float = 0.5,
    position_weight: float = 1.0,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    pred_deltas = predictions["pred_deltas"]
    pred_positions = predictions["pred_positions"]
    _validate_loss_shapes(pred_deltas, pred_positions, gt_deltas, gt_positions, future_mask)

    coord_mask = future_mask.to(device=pred_deltas.device, dtype=torch.bool).unsqueeze(-1)
    coord_mask = coord_mask.expand_as(pred_deltas)
    valid_count = coord_mask.sum()

    if valid_count.item() == 0:
        zero = (pred_deltas.sum() + pred_positions.sum()) * 0.0
        return {
            "loss": zero,
            "delta_loss": zero,
            "position_loss": zero,
            "valid_count": valid_count,
        }

    delta_loss = F.smooth_l1_loss(
        pred_deltas[coord_mask],
        gt_deltas.to(device=pred_deltas.device, dtype=pred_deltas.dtype)[coord_mask],
        beta=beta,
        reduction="mean",
    )
    position_loss = F.smooth_l1_loss(
        pred_positions[coord_mask],
        gt_positions.to(device=pred_positions.device, dtype=pred_positions.dtype)[coord_mask],
        beta=beta,
        reduction="mean",
    )
    total_loss = delta_weight * delta_loss + position_weight * position_loss

    return {
        "loss": total_loss,
        "delta_loss": delta_loss,
        "position_loss": position_loss,
        "valid_count": valid_count,
    }


def _validate_loss_shapes(
    pred_deltas: torch.Tensor,
    pred_positions: torch.Tensor,
    gt_deltas: torch.Tensor,
    gt_positions: torch.Tensor,
    future_mask: torch.Tensor,
) -> None:
    expected_prediction_shape = pred_deltas.shape

    if pred_deltas.dim() != 4 or expected_prediction_shape[-1] != 2:
        raise ValueError("pred_deltas must have shape [B, N, T_future, 2]")
    if pred_positions.shape != expected_prediction_shape:
        raise ValueError("pred_positions must have the same shape as pred_deltas")
    if gt_deltas.shape != expected_prediction_shape:
        raise ValueError("gt_deltas must have the same shape as pred_deltas")
    if gt_positions.shape != expected_prediction_shape:
        raise ValueError("gt_positions must have the same shape as pred_deltas")
    if future_mask.shape != expected_prediction_shape[:3]:
        raise ValueError("future_mask must have shape [B, N, T_future]")
