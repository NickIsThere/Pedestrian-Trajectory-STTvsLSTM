"""
Social LSTM model: per-pedestrian LSTMs with social pooling (Alahi et al., CVPR 2016).
No attention, no graph networks—just LSTM + occupancy grid pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from App.model.config import ModelConfig

_STT_DEFAULTS = ModelConfig()


class SocialPooling(nn.Module):
    """
    Main writer: Keez Cuijpers
    Reviewer: 
    Contributors: 
    
    Social pooling: groups nearby pedestrian hidden states into occupancy grid.
    Uses vectorized GPU operations for efficiency.
    """

    def __init__(
        self,
        hidden_size: int,
        grid_size: int = 4,
        pool_type: str = "max",
    ):
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 
        
        Args:
            hidden_size: LSTM hidden state dimension
            grid_size: Side length of occupancy grid (grid_size x grid_size)
            pool_type: "max" or "avg" for pooling operation
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.pool_type = pool_type
        self.grid_cells = grid_size * grid_size

    def forward(
        self,
        positions: torch.Tensor,  # (B, max_peds, 2) normalized [0, 1]
        hidden_states: torch.Tensor,  # (B, max_peds, hidden_size)
        masks: torch.Tensor,  # (B, max_peds) binary mask
    ) -> torch.Tensor:
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 
        
        Pool hidden states into occupancy grid around each pedestrian.
        
        Returns:
            (B, max_peds, grid_size * grid_size * hidden_size)
        """
        B, max_peds, _ = positions.shape
        device = positions.device

        # Prepare output tensor
        pooled = torch.zeros(
            B, max_peds, self.grid_cells, self.hidden_size,
            device=device, dtype=hidden_states.dtype
        )

        # Compute pairwise relative positions r_ij = pos_j - pos_i  -> (B, max_peds, max_peds, 2)
        rel = positions.unsqueeze(1) - positions.unsqueeze(2)  # (B, max_peds, max_peds, 2) where index [b,i,j]=pos_j-pos_i

        # Neighborhood mask: only consider neighbors within +/-0.5 normalized units
        neigh_mask = (rel.abs() <= 0.5).all(dim=-1)  # (B, max_peds, max_peds)

        # Combine with presence masks and exclude self
        mask_i = masks.unsqueeze(2)  # (B, max_peds, 1)
        mask_j = masks.unsqueeze(1)  # (B, 1, max_peds)
        present_mask = (mask_i.bool() & mask_j.bool())  # (B, max_peds, max_peds)
        # exclude self (i == j)
        idxs = torch.arange(max_peds, device=device)
        neq_self = idxs.unsqueeze(0) != idxs.unsqueeze(1)  # (max_peds, max_peds)
        neq_self = neq_self.unsqueeze(0)  # (1, max_peds, max_peds)

        valid_mask = neigh_mask & present_mask & neq_self

        # Compute cell indices for relative positions
        # Map r in [-0.5,0.5] to cell coords in [0, grid_size-1]
        cell_coords = ((rel + 0.5) * self.grid_size).floor().long()  # (B, max_peds, max_peds, 2)
        cell_x = cell_coords[..., 0]
        cell_y = cell_coords[..., 1]
        # clamp to valid range
        cell_x = cell_x.clamp(0, self.grid_size - 1)
        cell_y = cell_y.clamp(0, self.grid_size - 1)
        cell_idx_mat = cell_y * self.grid_size + cell_x  # (B, max_peds, max_peds)

        # Hidden states for neighbors expanded for broadcasting
        hid_j = hidden_states.unsqueeze(1)  # (B, 1, max_peds, hidden)
        hid_j = hid_j.expand(-1, max_peds, -1, -1)  # (B, max_peds, max_peds, hidden)

        for c in range(self.grid_cells):
            # boolean mask of neighbors that fall into cell c
            cell_mask = (cell_idx_mat == c) & valid_mask  # (B, max_peds, max_peds)
            if cell_mask.any():
                cell_mask_f = cell_mask.unsqueeze(-1)  # (B, max_peds, max_peds, 1)
                if self.pool_type == "max":
                    # set non-members to large negative so they don't affect max
                    neg_inf = torch.finfo(hidden_states.dtype).min
                    masked = torch.where(cell_mask_f, hid_j, torch.tensor(neg_inf, device=device, dtype=hidden_states.dtype))
                    pooled_cell = masked.max(dim=2)[0]  # (B, max_peds, hidden)
                    # replace neg_inf rows (no members) with zeros
                    no_members = (~cell_mask.any(dim=2)).unsqueeze(-1)
                    pooled_cell = torch.where(no_members, torch.zeros_like(pooled_cell), pooled_cell)
                else:  # avg
                    masked = hid_j * cell_mask_f.to(hidden_states.dtype)
                    sum_hidden = masked.sum(dim=2)  # (B, max_peds, hidden)
                    counts = cell_mask.sum(dim=2).clamp(min=1).unsqueeze(-1).to(hidden_states.dtype)
                    pooled_cell = sum_hidden / counts

                pooled[:, :, c, :] = pooled_cell

        pooled_flat = pooled.reshape(B, max_peds, self.grid_cells * self.hidden_size)
        return pooled_flat


class SocialLSTM(nn.Module):
    """
    Main writer: Keez Cuijpers
    Reviewer: 
    Contributors: 
    
    Social LSTM: per-pedestrian encoders with social pooling.
    Predicts future relative displacements from observation windows.
    """

    def __init__(
        self,
        obs_len: int = _STT_DEFAULTS.lookback,
        pred_len: int = _STT_DEFAULTS.future_steps,
        hidden_size: int = _STT_DEFAULTS.hidden_dim,
        embedding_dim: int = _STT_DEFAULTS.head_hidden_dim,
        grid_size: int = 4,
        input_dim: int = 2,  # x, y normalized positions
        output_dim: int = 2,  # dx, dy normalized displacements
    ):
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 
        
        Args:
            obs_len: Number of observation timesteps
            pred_len: Number of prediction timesteps
            hidden_size: LSTM hidden state size
            embedding_dim: Input embedding dimension
            grid_size: Social pooling grid size
            input_dim: Dimension of input features (position)
            output_dim: Dimension of output (displacement)
        """
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input embedding
        self.embedding = nn.Linear(input_dim, embedding_dim)

        # Encoder LSTM: processes observation window
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Social pooling
        self.social_pool = SocialPooling(
            hidden_size=hidden_size,
            grid_size=grid_size,
            pool_type="max",
        )

        # Pool→hidden bridge: map pooled grid to decoder hidden state
        pool_output_size = grid_size * grid_size * hidden_size
        self.pool_to_hidden = nn.Linear(pool_output_size, hidden_size)

        # Decoder LSTM: predicts future
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        obs_positions: torch.Tensor,  # (B, obs_len, max_peds, 2)
        obs_masks: torch.Tensor,  # (B, obs_len, max_peds)
        pred_positions: Optional[torch.Tensor] = None,  # (B, pred_len, max_peds, 2) for teacher forcing
        pred_masks: Optional[torch.Tensor] = None,  # (B, pred_len, max_peds)
        pred_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 
        
        Forward pass: encode observations with timestep-level social pooling, then decode with teacher forcing.
        
        Args:
            obs_positions: (B, obs_len, max_peds, 2) observed positions
            obs_masks: (B, obs_len, max_peds) observation masks
            pred_positions: (B, pred_len, max_peds, 2) ground truth for teacher forcing (optional)
            pred_masks: (B, pred_len, max_peds) prediction masks for masking (optional)
            pred_len: Override prediction length (default: self.pred_len)
        """
        if pred_len is None:
            pred_len = self.pred_len

        B, obs_steps, max_peds, _ = obs_positions.shape
        use_teacher_forcing = pred_positions is not None and pred_masks is not None
        device = obs_positions.device

        # Embed and encode observations with persistent temporal state.
        embedded_obs = self.embedding(obs_positions)  # (B, obs_len, max_peds, embedding_dim)

        h_enc = torch.zeros(1, B * max_peds, self.hidden_size, device=device, dtype=obs_positions.dtype)
        c_enc = torch.zeros_like(h_enc)

        h_history = []
        for t in range(obs_steps):
            obs_t = obs_positions[:, t, :, :]  # (B, max_peds, 2)
            mask_t = obs_masks[:, t, :]  # (B, max_peds)
            emb_t = embedded_obs[:, t, :, :]  # (B, max_peds, embedding_dim)

            emb_flat = emb_t.reshape(B * max_peds, 1, self.embedding_dim)
            _, (h_raw, c_enc) = self.encoder(emb_flat, (h_enc, c_enc))
            h_raw = h_raw.squeeze(0).reshape(B, max_peds, self.hidden_size)

            pooled_t = self.social_pool(obs_t, h_raw, mask_t)
            h_social = torch.tanh(self.pool_to_hidden(pooled_t))  # (B, max_peds, hidden_size)

            present = mask_t.unsqueeze(-1).to(dtype=h_social.dtype)
            h_merged = h_raw + h_social * present
            h_history.append(h_merged)

            h_enc = h_merged.reshape(1, B * max_peds, self.hidden_size)

        # Use final hidden state as decoder initial state
        h_dec = h_history[-1].reshape(B * max_peds, self.hidden_size)  # (B*max_peds, hidden_size)
        c_dec = torch.zeros_like(h_dec)  # (B*max_peds, hidden_size)
        
        # Seed decoder with last observed position embedding (not zeros!)
        last_obs_pos = obs_positions[:, -1, :, :]  # (B, max_peds, 2)
        decoder_input = self.embedding(last_obs_pos).reshape(B * max_peds, 1, self.embedding_dim)
        
        predictions = []
        cumulative_pos = last_obs_pos.clone()  # Track cumulative positions for predictions
        
        for step in range(pred_len):
            # Decoder LSTM step
            dec_out, (h_dec_new, c_dec_new) = self.decoder(
                decoder_input, (h_dec.unsqueeze(0), c_dec.unsqueeze(0))
            )
            # dec_out: (B*max_peds, 1, hidden_size)
            pred_displacement = self.output_proj(dec_out.squeeze(1))  # (B*max_peds, 2)
            predictions.append(pred_displacement)
            
            h_dec = h_dec_new.squeeze(0)
            c_dec = c_dec_new.squeeze(0)
            
            # Prepare next decoder input: use teacher forcing if available, else use prediction
            if use_teacher_forcing:
                next_pos = pred_positions[:, step, :, :]  # (B, max_peds, 2) ground truth
            else:
                # Accumulate predictions to get absolute positions
                pred_disp_reshaped = pred_displacement.reshape(B, max_peds, 2)
                next_pos = cumulative_pos + pred_disp_reshaped
                cumulative_pos = next_pos.clone()
            
            decoder_input = self.embedding(next_pos).reshape(B * max_peds, 1, self.embedding_dim)

        # Stack predictions: (pred_len, B*max_peds, 2)
        predictions = torch.stack(predictions, dim=0)
        
        # Reshape back to (B, pred_len, max_peds, 2)
        predictions = predictions.reshape(pred_len, B, max_peds, 2).permute(1, 0, 2, 3)
        
        return predictions

    def get_parameter_count(self) -> int:
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 
        
        Return total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
