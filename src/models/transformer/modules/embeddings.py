import math
from typing import Optional

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Map token IDs to learned vectors scaled by sqrt(d_model)."""

    def __init__(
        self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None,
        *, freeze: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model)
        self.set_frozen(freeze)

    def set_frozen(self, freeze: bool) -> None:
        """Toggle training and discard any pending gradient when freezing."""
        weight = self.embedding.weight
        weight.requires_grad_(not freeze)
        if freeze:
            weight.grad = None

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding(token_ids) * self.scale


class PositionalEmbedding(nn.Module):
    """Return fixed sinusoidal vectors for token IDs shaped (..., seq_len)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        if d_model < 1 or max_len < 1:
            raise ValueError("d_model and max_len must be positive")

        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        even_dims = torch.arange(0, d_model, 2, dtype=torch.float32)

        factors = torch.exp(-math.log(10000.0) * even_dims / d_model)
        angles = positions @ factors.unsqueeze(0)  # (3, 1) @ (1, 2) → (3, 2)

        encoding = torch.empty(max_len, d_model, dtype=angles.dtype)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, : d_model // 2].cos()
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        seq_len = token_ids.size(-1)
        if seq_len > self.encoding.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.encoding.size(0)}")
        return self.encoding[:seq_len].expand(*token_ids.shape, -1)


class TransformerEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None,
        *, freeze: bool = False,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx, freeze=freeze)
        self.positional_embedding = PositionalEmbedding(d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.token_embedding(token_ids) + self.positional_embedding(token_ids)


## --------- TEST --------- ##
if __name__ == "__main__":
    torch.manual_seed(0)
    token_ids = torch.tensor([
        [4, 7, 9],
        [7, 4, 0]
    ])
    model = TransformerEmbedding(vocab_size=10, d_model=4, padding_idx=0)

    with torch.no_grad():
        tokens = model.token_embedding(token_ids)
        positions = model.positional_embedding(token_ids)
        output = model(token_ids)

    assert tokens.shape == positions.shape == output.shape == (2, 3, 4)
    torch.testing.assert_close(tokens[0, 0], tokens[1, 1])  # Same token ID (4)
    torch.testing.assert_close(tokens[1, 2], torch.zeros(4))  # Padding token
    torch.testing.assert_close(positions[0], positions[1])
    torch.testing.assert_close(positions[0, 0], torch.tensor([0., 1., 0., 1.]))
    assert not torch.equal(positions[0, 0], positions[1, 1])
    torch.testing.assert_close(output, tokens + positions)

    print("token_ids:", token_ids.tolist())
    print("output.shape:", tuple(output.shape))
    print("token 4 + position 0:", output[0, 0].tolist())
    print("token 4 + position 1:", output[1, 1].tolist())

    # A trainable output layer keeps backward valid even with a frozen embedding.
    head = nn.Linear(4, 1, bias=False)
    optimizer = torch.optim.SGD([*model.parameters(), *head.parameters()], lr=0.1)
    print("Before training, head.weight.grad:", head.weight.grad)
    weight = model.token_embedding.embedding.weight
    print("Before training, embedding.weight.grad:", weight.grad)
    for freeze in (False, True, False):
        model.token_embedding.set_frozen(freeze)
        optimizer.zero_grad(set_to_none=True)
        before = weight.detach().clone()
        loss = head(model(token_ids)).mean()
        loss.backward()

        assert head.weight.grad is not None
        if freeze:
            assert weight.grad is None
        else:
            assert weight.grad is not None and weight.grad[4].abs().sum() > 0
            torch.testing.assert_close(weight.grad[0], torch.zeros_like(weight[0]))
            print(f"freeze={freeze}, token 4 grad={weight.grad[4].tolist()}")

        optimizer.step()
        if freeze:
            assert torch.equal(weight, before)
        else:
            assert not torch.equal(weight[4], before[4])
            assert torch.equal(weight[0], before[0])
        print(f"freeze={freeze}, token 4 updated={not torch.equal(weight[4], before[4])}")

    print("All embedding checks passed.")
