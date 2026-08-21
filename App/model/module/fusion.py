import torch.nn as nn

"""
    The crowd context fusion module
    Noah Nuelandt i6375705
    - retouches in masking: Nick Grebe i6377605
"""
class CrowdContextFusion(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x, mask=None):
        """
        input:
        x: [B, N, T, feature_dim]
        mask: [B, N], True for valid pedestrians
        output:
        x_fused: [B, N, T, feature_dim]
        """

        if mask is None:
            valid = x.new_ones(x.size(0), x.size(1), 1, 1)
        else:
            valid = mask.to(device=x.device).ne(0).to(dtype=x.dtype).view(x.size(0), x.size(1), 1, 1)

        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        context = (x * valid).sum(dim=1, keepdim=True) / denom
        out = self.norm(x + context)
        return out * valid
