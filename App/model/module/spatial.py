import torch
import torch.nn as nn

class SpatialInteractionBlock(nn.Module):
    """
    Lightweight masked k-NN spatial interaction block.

    Inputs use True/1 masks for valid agents and observations. Invalid output
    slots are zeroed before returning.
    
    nick grebe - i6377605
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 64,
        edge_dim: int = 8,
        k_neighbors: int = 3,
    ):
        super().__init__()
        if k_neighbors <= 0:
            raise ValueError("You have a typo, what even are negative neighbors!!")

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.k_neighbors = k_neighbors

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, hidden_dim),
        )
        self.edge_score = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        z0: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        speed: torch.Tensor,
        heading_sc: torch.Tensor,
        agent_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z0:[B, N, T, hidden_dim]
            positions: [B, N, T, 2]
            velocities: [B, N, T, 2]
            speed: [B, N, T, 1]
            heading_sc: [B, N, T, 2]
            agent_mask: [B, N], True for valid pedestrian slots
            time_mask: [B, N, T], True for valid observations

        Returns:
            Tensor with shape [B, N, T, hidden_dim].
        """
        self._validate_inputs(z0, positions, velocities, speed, heading_sc, agent_mask, time_mask)

        valid_obs = self._valid_observation_mask(agent_mask, time_mask, z0.device)
        if z0.size(1) == 0 or z0.size(2) == 0:
            return z0 * valid_obs.unsqueeze(-1).to(dtype=z0.dtype)

        # this is where the attention happens, wohooo
        edge_features, neighbor_mask = self._build_edge_features(
            positions=positions,
            velocities=velocities,
            speed=speed,
            heading_sc=heading_sc,
            valid_obs=valid_obs,
            dtype=z0.dtype,
        )

        edge_embeddings = self.edge_mlp(edge_features)
        edge_embeddings = edge_embeddings * neighbor_mask.unsqueeze(-1).to(dtype=edge_embeddings.dtype)

        scores = self.edge_score(edge_embeddings)
        weights = self._masked_softmax(scores, neighbor_mask.unsqueeze(-1), dim=3)
        messages = (weights * edge_embeddings).sum(dim=3)
        messages = messages.permute(0, 2, 1, 3)

        output = self.norm(z0 + messages)
        return output * valid_obs.unsqueeze(-1).to(dtype=output.dtype)

    def _build_edge_features(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        speed: torch.Tensor,
        heading_sc: torch.Tensor,
        valid_obs: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions_bt = positions.permute(0, 2, 1, 3)
        velocities_bt = velocities.permute(0, 2, 1, 3)
        speed_bt = speed.permute(0, 2, 1, 3)
        heading_bt = heading_sc.permute(0, 2, 1, 3)
        valid_bt = valid_obs.permute(0, 2, 1)

        batch_size, time_steps, num_agents, _ = positions_bt.shape
        topk_count = min(self.k_neighbors, num_agents)

        neighbor_indices, neighbor_mask = self._build_knn_graph(positions_bt, valid_bt, topk_count)

        neighbor_positions = self._gather_neighbors(positions_bt, neighbor_indices)
        neighbor_velocities = self._gather_neighbors(velocities_bt, neighbor_indices)
        neighbor_speed = self._gather_neighbors(speed_bt, neighbor_indices)
        neighbor_heading = self._gather_neighbors(heading_bt, neighbor_indices)

        rel_pos = neighbor_positions - positions_bt.unsqueeze(3)
        rel_velocity = neighbor_velocities - velocities_bt.unsqueeze(3)
        rel_distance = rel_pos.norm(dim=-1, keepdim=True)
        rel_speed = neighbor_speed - speed_bt.unsqueeze(3)
        rel_heading = neighbor_heading - heading_bt.unsqueeze(3)

        edge_features = torch.cat(
            [rel_pos, rel_velocity, rel_distance, rel_speed, rel_heading],
            dim=-1,
        ).to(dtype=dtype)

        if topk_count < self.k_neighbors:
            pad_count = self.k_neighbors - topk_count
            feature_pad = edge_features.new_zeros(
                batch_size,
                time_steps,
                num_agents,
                pad_count,
                self.edge_dim,
            )
            mask_pad = neighbor_mask.new_zeros(batch_size, time_steps, num_agents, pad_count)
            edge_features = torch.cat([edge_features, feature_pad], dim=3)
            neighbor_mask = torch.cat([neighbor_mask, mask_pad], dim=3)

        return edge_features * neighbor_mask.unsqueeze(-1).to(dtype=edge_features.dtype), neighbor_mask

    def _build_knn_graph(
        self,
        positions_bt: torch.Tensor,
        valid_bt: torch.Tensor,
        topk_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_agents = positions_bt.size(2)
        pairwise_distance = torch.cdist(positions_bt, positions_bt)

        valid_query = valid_bt.unsqueeze(3)
        valid_neighbor = valid_bt.unsqueeze(2)
        self_mask = torch.eye(num_agents, dtype=torch.bool, device=positions_bt.device).view(
            1,
            1,
            num_agents,
            num_agents,
        )
        pair_mask = valid_query & valid_neighbor & ~self_mask

        masked_distance = pairwise_distance.masked_fill(~pair_mask, torch.inf)
        _, neighbor_indices = torch.topk(masked_distance, k=topk_count, dim=3, largest=False)
        neighbor_mask = torch.gather(pair_mask, dim=3, index=neighbor_indices)

        return neighbor_indices, neighbor_mask

    @staticmethod
    def _gather_neighbors(values_bt: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
        _, _, _, channels = values_bt.shape
        gather_source = values_bt.unsqueeze(2).expand(-1, -1, values_bt.size(2), -1, -1)
        gather_index = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, -1, channels)
        return torch.gather(gather_source, dim=3, index=gather_index)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.to(device=scores.device, dtype=torch.bool)
        masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked_scores, dim=dim)
        return weights * mask.to(dtype=weights.dtype)

    @staticmethod
    def _valid_observation_mask(
        agent_mask: torch.Tensor,
        time_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        return agent_mask.to(device=device, dtype=torch.bool).unsqueeze(-1) & time_mask.to(
            device=device,
            dtype=torch.bool,
        )

    def _validate_inputs(
        self,
        z0: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        speed: torch.Tensor,
        heading_sc: torch.Tensor,
        agent_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> None:
        if z0.dim() != 4:
            raise ValueError("z0 must have shape [B, N, T, hidden_dim]")

        batch_size, num_agents, time_steps, hidden_dim = z0.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"z0 last dimension must be {self.hidden_dim}, spatial block got {hidden_dim}")

        expected_2d = (batch_size, num_agents, time_steps, 2)
        expected_1d = (batch_size, num_agents, time_steps, 1)
        if positions.shape != expected_2d:
            raise ValueError(f"positions must have shape {expected_2d}, spatial block got {tuple(positions.shape)}")
        if velocities.shape != expected_2d:
            raise ValueError(f"velocities must have shape {expected_2d}, spatial block got {tuple(velocities.shape)}")
        if speed.shape != expected_1d:
            raise ValueError(f"speed must have shape {expected_1d}, spatial block got {tuple(speed.shape)}")
        if heading_sc.shape != expected_2d:
            raise ValueError(f"heading_sc must have shape {expected_2d}, spatial block got {tuple(heading_sc.shape)}")
        if agent_mask.shape != (batch_size, num_agents):
            raise ValueError(
                f"agent_mask must have shape {(batch_size, num_agents)}, spatial block got {tuple(agent_mask.shape)}"
            )
        if time_mask.shape != (batch_size, num_agents, time_steps):
            raise ValueError(
                f"time_mask must have shape {(batch_size, num_agents, time_steps)}, spatial block got {tuple(time_mask.shape)}"
            )
