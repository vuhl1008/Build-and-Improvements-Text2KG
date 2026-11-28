import torch
import torch.nn as nn

if __package__:
    from .attention import MultiHeadAttention
    from .layers import FeedForward, ResidualConnection
else:
    from attention import MultiHeadAttention
    from layers import FeedForward, ResidualConnection

class EncoderBlock(nn.Module):
    """A single encoder block for a Transformer model."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0, *, freeze: bool = False):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout, freeze=freeze)
        self.residual1 = ResidualConnection(d_model, freeze=freeze)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout, freeze=freeze)
        self.residual2 = ResidualConnection(d_model, freeze=freeze)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return (batch, seq_len, d_model). True in the (batch, seq_len) mask skips a key."""
        attn_output = self.self_attention(x, key_padding_mask)
        x = self.residual1(x, attn_output)
        ff_output = self.feed_forward(x)
        x = self.residual2(x, ff_output)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    key_padding_mask = torch.tensor([[False, False, True], [False, True, True]])
    encoder_block = EncoderBlock(d_model=4, num_heads=2, d_ff=8, dropout=0.1)
    encoder_block.eval()

    with torch.no_grad():
        attention_output = encoder_block.self_attention(x, key_padding_mask)
        residual1_output = encoder_block.residual1(x, attention_output)
        ff_output = encoder_block.feed_forward(residual1_output)
        residual2_output = encoder_block.residual2(residual1_output, ff_output)
        output = encoder_block(x, key_padding_mask)

    assert output.shape == (2, 3, 4)
    torch.testing.assert_close(residual2_output, output)
    print("output.shape:", tuple(output.shape))

    print("x:", x)
    print("key_padding_mask:", key_padding_mask)
    print("attention output:", attention_output)
    print("residual1 output:", residual1_output)
    print("feed-forward output:", ff_output)
    print("residual2 output:", residual2_output)

    for name, parameter in encoder_block.named_parameters():
        print(f"{name}: {tuple(parameter.shape)}")
