import torch.nn as nn

"""
    The temporal patching module
    Noah Nuelandt i6375705
"""
class TemporalPatching(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, stride=3)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        input:
        x: [B,N,T,feature_dim]
        output:
        x: [B,N,T/3,feature_dim]
        """
        B, N, T, C = x.shape
        
        # [B, N, T, feature_dim] -> [B*N, T, feature_dim] -> [B*N, feature_dim, T]
        x = x.reshape(B * N, T, C).transpose(1, 2)
        
        x = self.conv(x) # Slide over the T dimension
        x = self.gelu(x)
        
        # Transpose back
        x = x.transpose(1, 2).reshape(B, N, -1, C)
        
        x = self.norm(x)
        return x