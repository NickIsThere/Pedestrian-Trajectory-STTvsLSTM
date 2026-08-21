import torch.nn as nn

class TrajectoryTransformerEncoder(nn.Module):
    """
        The encoder layer of the Transformer model
        Neo Deward I6382733
    """

    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, num_layers=3, dropout=0.1):
        """
            Definition of the encoder layer
            input:
            embed_dim: dimension of the embedding
            num_heads: number of heads
            ff_dim: dimension of the feedforward layer
            num_layers: number of layers
            dropout: dropout rate
            Neo Deward I6382733
        """
        super().__init__()


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, src, mask=None):
        """
            input:
            src:[B*N,15,128]
            mask:[B*N,15]
            Neo Deward I6382733
        """
        return self.encoder(src, src_key_padding_mask=mask)