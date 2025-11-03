# Step 04: Feed-forward network

<div class="note">
    Learn to build the feed-forward network (MLP) that processes information after attention in each transformer block.
</div>

## Implementing the MLP

In this step you'll build the `MLP` class, a two-layer feed-forward network that appears after the attention mechanism in every transformer block. The MLP expands the embedding dimension by 4x (768 → 3072), applies a GELU activation function, then projects back to the original dimension (3072 → 768).

While attention aggregates information across tokens through weighted sums, the MLP adds crucial non-linearity through the GELU activation. This allows the model to learn complex patterns beyond what linear transformations can capture.

## Understanding the architecture

The MLP consists of two linear layers with GELU activation between them:

1. **Expansion layer (c_fc)**: Projects from embedding dimension (768) to intermediate size (3072 = 4×768). The 4× expansion ratio comes from the original Transformer paper and has been empirically validated across many architectures.
2. **GELU activation**: Applies smooth non-linear transformation
3. **Projection layer (c_proj)**: Projects from intermediate size back to embedding dimension (768)

GELU (Gaussian Error Linear Unit) is the activation function used in GPT-2. The `approximate="tanh"` parameter uses a tanh-based approximation instead of the exact Gaussian computation. At the time GPT-2 was implemented, the exact GELU was too slow, so the tanh approximation was used. While exact GELU is fast enough now, we use the approximation to match GPT-2's original implementation for weight compatibility.

The layer names `c_fc` and `c_proj` match the original GPT-2 checkpoint structure for weight loading compatibility.

<div class="note">
<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Linear layers**:
- [`Linear(in_features, out_features, bias=True)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear): Applies linear transformation `y = xW^T + b`

**GELU activation**:
- [`F.gelu(input, approximate="tanh")`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.gelu): Applies GELU activation with tanh approximation for faster computation

</div>

## Implementing the class

You'll implement the `MLP` class in several steps:

1. **Import required modules**: Import `functional as F`, `Tensor`, `Linear`, and `Module` from MAX libraries.

2. **Create expansion layer**: Use `Linear(embed_dim, intermediate_size, bias=True)` and store in `self.c_fc`.

3. **Create projection layer**: Use `Linear(intermediate_size, embed_dim, bias=True)` and store in `self.c_proj`.

4. **Apply expansion**: In the forward pass, apply `self.c_fc(hidden_states)` to expand the representation to intermediate size.

5. **Apply GELU**: Use `F.gelu(hidden_states, approximate="tanh")` for non-linear transformation.

6. **Apply projection**: Apply `self.c_proj(hidden_states)` to project back to original dimension and return the result.

**Implementation** (`step_04.py`):

```python
{{#include ../../steps/step_04.py}}
```

### Validation

Run `pixi run s04` to verify your implementation.

**Reference**: `solutions/solution_04.py`

**Next**: In [Step 05](./step_05.md), you'll implement token embeddings to convert discrete token IDs into continuous vector representations.
