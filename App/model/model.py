import torch
import torch.nn as nn

from .config import ModelConfig
from .layers.transformer import TrajectoryTransformerEncoder
from .module.fusion import CrowdContextFusion
from .module.regression import RegressionHead
from .module.spatial import SpatialInteractionBlock
from .module.stem import MotionFeatureStem
from .module.temporal import TemporalPatching
from .module.tokens import FutureTokenBlock


class TrajectoryTransformer(nn.Module):
    """
    Our unified forward pass for the SP Transformer

    Nick Grebe i6377605
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()

        self.stem = MotionFeatureStem(self.config)
        self.spatial = SpatialInteractionBlock(
            hidden_dim=self.config.hidden_dim,
            k_neighbors=self.config.k_neighbors,
        )
        self.temporal = TemporalPatching(feature_dim=self.config.hidden_dim)
        self.crowd_fusion = CrowdContextFusion(feature_dim=self.config.hidden_dim)
        self.future_tokens = FutureTokenBlock(self.config)
        self.encoder = TrajectoryTransformerEncoder(
            embed_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            ff_dim=self.config.ff_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.regression_head = RegressionHead(
            embed_dim=self.config.hidden_dim,
            hidden_dim=self.config.head_hidden_dim,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        speed: torch.Tensor,
        heading_sc: torch.Tensor,
        agent_mask: torch.Tensor,
        time_mask: torch.Tensor,
        future_mask: torch.Tensor | None = None,
        last_obs_pos: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_forward_inputs(x, positions, velocities, speed, heading_sc, agent_mask, time_mask)

        agent_mask = agent_mask.to(device=x.device, dtype=torch.bool)
        time_mask = time_mask.to(device=x.device, dtype=torch.bool)

        z0 = self.stem(x, time_mask)
        z_spatial = self.spatial(z0, positions, velocities, speed, heading_sc, agent_mask, time_mask)
        h = self.temporal(z_spatial)
        h_crowd = self.crowd_fusion(h, agent_mask)

        history_token_mask = self._history_token_mask(time_mask, agent_mask)
        z_in, encoder_mask = self.future_tokens(h_crowd, agent_mask, history_token_mask)
        encoder_output = self._encode_valid_agents(z_in, encoder_mask, agent_mask)
        future_tokens = encoder_output[:, :, -self.config.future_steps :, :]

        pred_deltas = self.regression_head(future_tokens)
        valid_agents = agent_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=pred_deltas.dtype)
        pred_deltas = pred_deltas * valid_agents

        if last_obs_pos is None:
            last_obs_pos = self._last_observed_position(positions, time_mask, agent_mask)
        else:
            last_obs_pos = last_obs_pos.to(device=pred_deltas.device, dtype=pred_deltas.dtype)

        pred_positions = last_obs_pos + torch.cumsum(pred_deltas, dim=2)
        pred_positions = pred_positions * valid_agents

        return {
            "pred_deltas": pred_deltas,
            "pred_positions": pred_positions,
            "encoder_output": encoder_output,
            "future_tokens": future_tokens,
        }

    def _encode_valid_agents(
        self,
        z_in: torch.Tensor,
        encoder_mask: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_agents, token_count, hidden_dim = z_in.shape
        z_flat = z_in.reshape(batch_size * num_agents, token_count, hidden_dim)
        mask_flat = encoder_mask.reshape(batch_size * num_agents, token_count)
        valid_flat = agent_mask.reshape(batch_size * num_agents)

        encoded_flat = z_flat.new_zeros(batch_size * num_agents, token_count, hidden_dim)
        if valid_flat.any():
            encoded_flat[valid_flat] = self.encoder(z_flat[valid_flat], mask_flat[valid_flat])

        return encoded_flat.reshape(batch_size, num_agents, token_count, hidden_dim)

    def _history_token_mask(self, time_mask: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_agents, time_steps = time_mask.shape
        if time_steps != self.config.lookback:
            raise ValueError(f"time_mask must have {self.config.lookback} observed steps")

        patch_count = self.config.lookback // self.config.patch_size
        usable_steps = patch_count * self.config.patch_size
        patches = time_mask[:, :, :usable_steps].reshape(
            batch_size,
            num_agents,
            patch_count,
            self.config.patch_size,
        )
        history_mask = patches.any(dim=-1)
        return history_mask & agent_mask.to(device=time_mask.device, dtype=torch.bool).unsqueeze(-1)

    def _last_observed_position(
        self,
        positions: torch.Tensor,
        time_mask: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_time = time_mask.to(device=positions.device, dtype=torch.bool)
        time_indices = torch.arange(positions.size(2), device=positions.device).view(1, 1, -1)
        masked_indices = torch.where(valid_time, time_indices, torch.zeros_like(time_indices))
        last_indices = masked_indices.max(dim=2).values

        gather_index = last_indices.view(positions.size(0), positions.size(1), 1, 1).expand(-1, -1, 1, 2)
        last_pos = torch.gather(positions, dim=2, index=gather_index)
        has_history = valid_time.any(dim=2) & agent_mask.to(device=positions.device, dtype=torch.bool)
        return last_pos * has_history.unsqueeze(-1).unsqueeze(-1).to(dtype=positions.dtype)

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        speed: torch.Tensor,
        heading_sc: torch.Tensor,
        agent_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> None:
        if x.dim() != 4:
            raise ValueError("x must have shape [B, N, T_obs, input_dim]")

        batch_size, num_agents, time_steps, input_dim = x.shape
        if time_steps != self.config.lookback:
            raise ValueError(f"x must have {self.config.lookback} observed steps")
        if input_dim != self.config.input_dim:
            raise ValueError(f"x last dimension must be {self.config.input_dim}")

        expected_2d = (batch_size, num_agents, time_steps, 2)
        expected_1d = (batch_size, num_agents, time_steps, 1)
        if positions.shape != expected_2d:
            raise ValueError(f"positions must have shape {expected_2d}")
        if velocities.shape != expected_2d:
            raise ValueError(f"velocities must have shape {expected_2d}")
        if speed.shape != expected_1d:
            raise ValueError(f"speed must have shape {expected_1d}")
        if heading_sc.shape != expected_2d:
            raise ValueError(f"heading_sc must have shape {expected_2d}")
        if agent_mask.shape != (batch_size, num_agents):
            raise ValueError(f"agent_mask must have shape {(batch_size, num_agents)}")
        if time_mask.shape != (batch_size, num_agents, time_steps):
            raise ValueError(f"time_mask must have shape {(batch_size, num_agents, time_steps)}")


HTPModel = TrajectoryTransformer # HTP model is just to make everything backwards compatible with the model that was here before!
