from torch import Tensor, nn

from modules.encoder_block import EncoderBlock


class Encoder(nn.Module):
    """Stack encoder blocks over embedded source tokens."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, num_layers: int,
        dropout: float = 0.0, *, freeze: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.layers = nn.ModuleList(
            EncoderBlock(d_model, num_heads, d_ff, dropout, freeze=freeze)
            for _ in range(num_layers)
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """Encode (batch, seq_len, d_model), ignoring keys marked True in the mask."""
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x
