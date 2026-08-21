from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_dim: int = 9
    hidden_dim: int = 128
    head_hidden_dim: int = 64
    num_heads: int = 4
    ff_dim: int = 256
    num_layers: int = 3
    k_neighbors: int = 3
    lookback: int = 15
    patch_size: int = 3
    future_steps: int = 10
    trajectory_stride: int = 5
    window_start_stride: int = 1
    stem_dropout: float = 0.05
    dropout: float = 0.1
