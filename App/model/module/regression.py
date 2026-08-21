import torch.nn as nn

class RegressionHead(nn.Module):
    """
       The regression head
        Neo Deward I6382733
    """
    def __init__(self,embed_dim=128, hidden_dim=64,dropout=0.1):
        """
            Definition of the encoder layer
            input:
            embed_dim: dimension of the embedding
            hidden_dim: dimension of the hidden layer
            dropout: dropout rate
            Neo Deward I6382733
        """
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(embed_dim,hidden_dim)
            ,nn.GELU()
            ,nn.Dropout(dropout)
            ,nn.Linear(hidden_dim,2))

    def forward(self,y_future):
        """
            input:
            y_future: [B,N,10,128]
            Neo Deward I6382733
            Small edit for rest of model compatability Nick grebe i6377605
        """
        return self.net(y_future)
