# Step 07: Query/Key/Value projections (single head)

<div class="note">
    Learn to implement Q/K/V projection layers that transform embeddings for attention computation.
</div>

## Implementing Q/K/V projections

In this step you'll implement the Q/K/V projections for attention. These linear layers transform input embeddings into three different representations:

- Query: what am I looking for?
- Key: what do I contain?
- Value: what information do I carry?

GPT-2 uses a single combined linear layer called `c_attn` that projects from embedding dimension (768) to 3 times that size (2304), then splits the output into separate Q, K, and V tensors. This combined projection is more efficient than three separate `Linear(768, 768)` layers. A single matrix multiplication with a larger weight matrix is faster due to better hardware utilization and reduced memory overhead.

Projections allow the model to learn transformations that make attention patterns easier to detect. The same input embedding gets projected into three different spaces (Q, K, V), each optimized for its role in the attention mechanism.

## Understanding the projections

This step implements single-head attention (simpler to understand before introducing multiple heads). The full GPT-2 uses 12 attention heads, which we'll add in [Step 09](./step_09.md).

**Key operations**:
- Combined projection: `Linear(n_embd, 3 * n_embd)` creates Q, K, V in one layer
- Split: Divide output into three equal parts for Q, K, V
- Each part has shape: [batch, seq_length, n_embd]

<div class="note">
<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Linear projection**:
- [`Linear(in_features, out_features, bias=True)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear): Applies linear transformation

**Splitting tensor**:
- [`F.split(tensor, split_sizes, axis)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.split): Divides tensor along an axis into multiple parts

</div>

## Implementing the projections

You'll implement the Q/K/V projections in several steps:

1. **Import required modules**: Import `Linear`, `Module`, and `functional as F` from MAX libraries.

2. **Create combined projection**: Use `Linear(config.n_embd, 3 * config.n_embd, bias=True)` and store in `self.c_attn`.

3. **Project input**: Call `self.c_attn(x)` to get concatenated Q/K/V. Input shape: [batch, seq_length, n_embd]. Output shape: [batch, seq_length, 3 * n_embd].

4. **Split into Q, K, V**: Use `F.split(qkv, [self.n_embd, self.n_embd, self.n_embd], axis=-1)` to divide the output into three equal parts.

**Implementation** (`step_07.py`):

```python
{{#include ../../steps/step_07.py}}
```

### Validation

Run `pixi run s07` to verify your implementation.

**Reference**: `solutions/solution_07.py`

**Next**: In [Step 08](./step_08.md), you'll implement the attention mechanism itself, computing attention scores from Q and K, applying causal masking, and using those scores to weight the values.
