import torch
import torch.nn as nn
from torch import Tensor, nn

class LayerNorm(nn.Module):
    """Layer normalization module with optional freezing of parameters."""
    def __init__(self, d_model: int, eps: float = 1e-5, freeze: bool = False) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)
        self.set_frozen(freeze)

    def set_frozen(self, freeze: bool) -> None:
        """Toggle gradients for the layer norm parameters while preserving gradients to the input."""
        self.requires_grad_(not freeze)
        if freeze:
            self.zero_grad(set_to_none=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer_norm(x)

class FeedForward(nn.Module):
    """Feed-forward network for a Transformer encoder layer."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, *, freeze: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.set_frozen(freeze)

    def set_frozen(self, freeze: bool) -> None:
        """Toggle gradients for the feed-forward parameters while preserving gradients to the input."""
        self.requires_grad_(not freeze)
        if freeze:
            self.zero_grad(set_to_none=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class ResidualConnection(nn.Module):
    """Residual connection with optional layer normalization."""
    def __init__(self, d_model: int, eps: float = 1e-5, *, freeze: bool = False):
        super().__init__()
        self.layer_norm = LayerNorm(d_model, eps=eps, freeze=freeze)

    def forward(self, x: Tensor, sublayer_output: Tensor) -> Tensor:
        return x + sublayer_output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    layer_norm = LayerNorm(d_model=4)
    feed_forward = FeedForward(d_model=4, d_ff=8, dropout=0.1)
    residual = ResidualConnection(d_model=4)

    with torch.no_grad():
        normed_x = layer_norm(x)
        ff_output = feed_forward(normed_x)
        output = residual(x, ff_output)

    assert normed_x.shape == ff_output.shape == output.shape == (2, 3, 4)
    print("LayerNorm output:", normed_x)
    print("FeedForward output:", ff_output)
    print("ResidualConnection output:", output)