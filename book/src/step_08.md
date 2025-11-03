# Step 08: Attention mechanism with causal masking

<div class="note">
    Learn to implement the core attention mechanism using scaled dot-product attention with causal masking.
</div>

## Implementing attention

In this step you'll implement the attention mechanism. Given query, key, and value tensors from Step 07, attention computes how much each position should "pay attention to" other positions, then creates weighted combinations of values. The process: compute similarity scores (Q @ K^T), scale by sqrt(d_k), apply causal mask to block future tokens, apply softmax to convert scores to probabilities, and multiply probabilities by values.

Causal masking enforces that each token can only attend to previous tokens. During generation, token N cannot see tokens N+1, N+2, etc. because they don't exist yet. Training with causal masking ensures the model sees the same information during training as it will during generation.

## Understanding scaled dot-product attention

The attention formula is:

```math
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

**Key steps**:
1. Compute similarity scores: `query @ key.transpose(-1, -2)`
2. Scale by `sqrt(d_k)` to prevent softmax saturation
3. Apply causal mask: add -∞ to future positions
4. Apply softmax: converts scores to probabilities
5. Weighted sum: `attn_weights @ value`

Scaling by sqrt(d_k) is critical. Without it, the dot products grow large as the embedding dimension increases, pushing softmax into regions with extremely small gradients (saturation). For d_k=64, typical dot products have variance around 64. Dividing by sqrt(64)=8 normalizes them to unit variance, keeping softmax in a responsive range.

The causal mask adds -∞ to attention scores for future positions. After softmax, e^(-∞) = 0, so future positions get zero attention weight.

<div class="note">
<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Causal masking**:
- [`F.band_part(mask, num_lower=None, num_upper=0, exclude=True)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.band_part): Creates upper-triangular matrix of -∞ values

**Softmax**:
- [`F.softmax(attn_weights)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.softmax): Converts scores to probabilities

</div>

## Implementing the mechanism

You'll implement attention in several steps:

1. **Import required modules**: Import `math`, `functional as F`, `Tensor`, `Device`, `DType`, `Dim`, and `DimLike` from MAX libraries.

2. **Implement causal_mask function**: Use `@F.functional` decorator, create -∞ constant tensor, broadcast to shape, and apply `F.band_part()` to create upper triangle mask.

3. **Compute attention scores**: Multiply query and transposed key: `query @ key.transpose(-1, -2)`. Shape: [..., seq_length, seq_length].

4. **Scale scores**: Divide by `math.sqrt(int(value.shape[-1]))` to prevent softmax saturation.

5. **Apply causal mask**: Create mask with `causal_mask()`, then add to scores: `attn_weights + mask`.

6. **Apply softmax and compute output**: Normalize with `F.softmax(attn_weights)`, then compute weighted sum: `attn_weights @ value`.

**Implementation** (`step_08.py`):

```python
{{#include ../../steps/step_08.py}}
```

### Validation

Run `pixi run s08` to verify your implementation.

**Reference**: `solutions/solution_08.py`

**Next**: In [Step 09](./step_09.md), you'll extend this single-head attention to multi-head attention, allowing the model to attend to different representation subspaces simultaneously.
