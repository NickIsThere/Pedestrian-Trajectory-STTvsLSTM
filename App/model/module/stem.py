from linecache import cache

import torch
import torch.nn as nn

# import sys
# import os
#
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# from config import ModelConfig


# made by Claire Bams i6402915

class MotionFeatureStem(nn.Module):


    # made by Claire Bams i6402915
    def __init__(self, config):
        super().__init__()

        self.fc1  = nn.Linear(config.input_dim, config.hidden_dim // 2) #(128 to 64, otherwise GELU unnecessary)
        self.act  = nn.GELU() #Adds non-linearity
        self.fc2  = nn.Linear(config.hidden_dim // 2,config.hidden_dim) #(64 to 128)
        self.norm = nn.LayerNorm(config.hidden_dim) # rescale value to a m=0 , var = 1
        self.drop = nn.Dropout(p=config.stem_dropout) # avoid overfitting



    # made by Claire Bams i6402915
    def dfmask(self, time_mask):
        # converts [B, N, 15] bool -> [B, N, 15, 1] float
        # unsqueeze adds the extra dimension, .float() turns True/False to 1.0/0.0
        return time_mask.unsqueeze(-1).float()


    # made by Claire Bams i6402915
    def forward(self, x, time_mask):
        """
        x         : [B, N, 15, 9]
        time_mask : [B, N, 15]    True = valid timestep

        returns z0 : [B, N, 15, 128]
        """
        # zero out invalid timesteps before first linear
        mask = self.dfmask(time_mask)   # [B, N, 15, 1]
        x    = x * mask                 # [B, N, 15, 9]

        x = self.fc1(x)                 # [B, N, 15, 64]
        x = self.act(x)                 # [B, N, 15, 64]
        x = self.fc2(x)                 # [B, N, 15, 128]
        x = self.norm(x)                # [B, N, 15, 128]
        x = self.drop(x)                # [B, N, 15, 128]

        return x
