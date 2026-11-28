import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """Batch-first multi-head attention with optional cross-attention context."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, *, freeze: bool = False):
        super().__init__()
        if d_model < 1 or num_heads < 1:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.set_frozen(freeze)

    def set_frozen(self, freeze: bool) -> None:
        """Toggle gradients for the attention parameters while preserving gradients to the input."""
        self.requires_grad_(not freeze)
        if freeze:
            self.zero_grad(set_to_none=True)

    def forward(
        self, x: Tensor, key_padding_mask: Tensor | None = None, *, context: Tensor | None = None
    ) -> Tensor:
        """Use x for queries and context (default x) for keys and values.

        Return (batch, query_len, d_model). True in the (batch, key_len) mask skips a key.
        """
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"x must have shape (batch, query_len, {self.d_model})")
        batch_size, query_len, _ = x.shape
        
        if context is not None and (
            context.ndim != 3 or context.size(0) != batch_size or context.size(-1) != self.d_model
        ):
            raise ValueError(f"context must have shape ({batch_size}, key_len, {self.d_model})")
        key_len = query_len if context is None else context.size(1)

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.shape != (batch_size, key_len):
                raise ValueError("key_padding_mask must be a boolean tensor of shape (batch, key_len)")

        if context is None:
            query, key, value = self.qkv_proj(x).chunk(3, dim=-1)
        else:
            # The packed projection stores Q first, followed by K and V.
            weight, bias = self.qkv_proj.weight, self.qkv_proj.bias
            query = F.linear(x, weight[:self.d_model], bias[:self.d_model])
            key_value = F.linear(context, weight[self.d_model:], bias[self.d_model:])
            key, value = key_value.chunk(2, dim=-1)

        query = query.reshape(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        compute_dtype = torch.promote_types(query.dtype, torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            # Each head computes a (query_len, key_len) matrix of scores.
            scores = query.to(compute_dtype) @ key.to(compute_dtype).transpose(-2, -1)
            scores = scores / math.sqrt(self.head_dim)

            if key_padding_mask is not None:
                padding = key_padding_mask[:, None, None, :]
                scores = scores.masked_fill(padding, torch.finfo(scores.dtype).min)

            weights = torch.softmax(scores, dim=-1)
            if key_padding_mask is not None:
                # A sequence containing only padding contributes zero at every key.
                weights = weights.masked_fill(padding, 0.0)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            heads = (weights @ value.to(compute_dtype)).to(value.dtype)

        merged = heads.transpose(1, 2).reshape(batch_size, query_len, self.d_model)
        return self.out_proj(merged)



## --------- TEST --------- ##
if __name__ == "__main__":
    if __package__:
        from .embeddings import TransformerEmbedding
    else:
        from embeddings import TransformerEmbedding

    torch.manual_seed(0)

    pad_id = 0
    token_ids = torch.tensor([
        [4, 7, 0],
        [2, 5, 9]
    ])
    embedding = TransformerEmbedding(vocab_size=10, d_model=4, padding_idx=pad_id)
    attention = MultiHeadAttention(d_model=4, num_heads=2)

    vectors = embedding(token_ids)
    output = attention(vectors, key_padding_mask=token_ids.eq(pad_id))
    assert output.shape == (2, 3, 4)

    # Compare three queries over five context tokens with PyTorch's attention.
    context = torch.randn(2, 5, 4)
    context_mask = torch.tensor([[False, False, False, True, True], [True, False, False, False, False]])
    reference = nn.MultiheadAttention(4, 2, batch_first=True)
    with torch.no_grad():
        reference.in_proj_weight.copy_(attention.qkv_proj.weight)
        reference.in_proj_bias.copy_(attention.qkv_proj.bias)
        reference.out_proj.load_state_dict(attention.out_proj.state_dict())
        expected, _ = reference(vectors, context, context, key_padding_mask=context_mask)
        actual = attention(vectors, context=context, key_padding_mask=context_mask)
    assert actual.shape == (2, 3, 4)
    torch.testing.assert_close(actual, expected)

    print("Before training, embedding.weight.grad:", embedding.token_embedding.embedding.weight.grad)
    print("Before training, attention.out_proj.weight.grad:", attention.out_proj.weight.grad)

    # Compare the real tokens with the same sequence run without padding.
    with torch.no_grad():
        unpadded = attention(embedding(torch.tensor([[4, 7]])))
    torch.testing.assert_close(output[0, :2], unpadded[0])

    # A loss on token 4 reaches token 7 through self-attention.
    output[0, 0].square().sum().backward()
    grad = embedding.token_embedding.embedding.weight.grad
    assert grad is not None and grad[7].abs().sum().item() > 0
    torch.testing.assert_close(grad[pad_id], torch.zeros_like(grad[pad_id]))

    print("After training, embedding.weight.grad:", grad)
    print("After training, attention.qkv_proj.weight.grad:", attention.qkv_proj.weight.grad)
    print("After training, attention.out_proj.weight.grad:", attention.out_proj.weight.grad)

    optimizer = torch.optim.SGD([*embedding.parameters(), *attention.parameters()], lr=0.01)
    for freeze in (True, False):
        attention.set_frozen(freeze)
        if freeze:
            assert all(p.grad is None for p in attention.parameters())
        optimizer.zero_grad(set_to_none=True)
        before = {name: p.detach().clone() for name, p in attention.named_parameters()}
        output = attention(embedding(token_ids), key_padding_mask=token_ids.eq(pad_id))
        output[0, 0].square().sum().backward()
        assert embedding.token_embedding.embedding.weight.grad[7].abs().sum() > 0
        assert all((p.grad is None) == freeze for p in attention.parameters())
        optimizer.step()
        if freeze:
            for name, p in attention.named_parameters():
                assert torch.equal(p, before[name])
        else:
            assert not torch.equal(attention.qkv_proj.weight, before["qkv_proj.weight"])
            assert not torch.equal(attention.out_proj.weight, before["out_proj.weight"])
        print(f"freeze={freeze}, attention gradients and optimizer update checked.")
    print("Embedding + multi-head attention test passed.")
