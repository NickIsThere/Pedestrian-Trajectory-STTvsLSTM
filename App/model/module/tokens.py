import torch
import torch.nn as nn

# look at 15 timesteps of the past ----> predict 10 timesteps of the future

# made by Claire Bams i6402915 retouched by Nick Grebe i6377605 for compatibility with the rest of the model
class FutureTokenBlock(nn.Module):

    # made by Claire Bams i6402915
    def __init__(self, config):
        super().__init__()

        n_patches = config.lookback // config.patch_size
        self.future_steps = config.future_steps

        self.future_tokens = nn.Parameter(torch.zeros(1, 1, config.future_steps, config.hidden_dim)) #(make blank for future [1, 1, 10, 128])
        self.history_positions = nn.Parameter(torch.zeros(1, 1, n_patches, config.hidden_dim))
        self.future_positions = nn.Parameter(torch.zeros(1, 1, config.future_steps, config.hidden_dim))#(all futur pos)

        nn.init.trunc_normal_(self.future_tokens,std=0.02)
        nn.init.trunc_normal_(self.history_positions,std=0.02)
        nn.init.trunc_normal_(self.future_positions,std=0.02)

    # made by Claire Bams i6402915
    def forward(self, h_crowd, agent_mask, history_token_mask):

        B, N, _, _ = h_crowd.shape
        agent_mask = agent_mask.to(device=h_crowd.device, dtype=torch.bool)
        history_token_mask = history_token_mask.to(device=h_crowd.device, dtype=torch.bool)

        h = h_crowd + self.history_positions.to(device=h_crowd.device, dtype=h_crowd.dtype) # [B, N, 5, 128]


        f = self.future_tokens.expand(B, N, -1, -1)  # [B, N, 10, 128]
        f = f + self.future_positions.to(device=h_crowd.device, dtype=h_crowd.dtype) # [B, N, 10, 128]

        # zero out fake agents
        valid = agent_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=h_crowd.dtype)
        h = h * valid
        f     = f * valid # [B, N, 10, 128]

        z_in = torch.cat([h, f], dim=2) # [B, N, 15, 128]

        # padding mask for transformer
        future_token_mask = agent_mask.unsqueeze(-1).expand(B, N, self.future_steps)
        token_mask = torch.cat([history_token_mask, future_token_mask], dim=2)
        encoder_mask = ~token_mask.reshape(B * N, token_mask.size(2))

        return z_in, encoder_mask

