# Step 09: Multi-head attention

<div class="note">
    Learn to extend single-head attention to multi-head attention, enabling the model to attend to different representation subspaces.
</div>

## Implementing multi-head attention

In this step you'll extend single-head attention to multi-head attention. Instead of one attention operation, you'll run 12 attention operations in parallel, each learning to focus on different patterns.

GPT-2 uses 12 heads with 768-dimensional embeddings. Each head operates on 768 ÷ 12 = 64 dimensions. This 64-dimensional head size strikes a balance: large enough to capture meaningful patterns, small enough to allow multiple specialized heads. The Q, K, V tensors are reshaped to split the embedding dimension across heads, then attention is computed for all heads in parallel. The outputs are concatenated back together.

Multiple heads let the model learn complementary attention strategies. Different heads can specialize in different types of relationships: one might attend to adjacent tokens, another to syntactic patterns, another to semantic similarity. This increases the model's capacity without dramatically increasing computation.

## Understanding multi-head attention

**Head Splitting**:
- Transform `[batch, seq_length, n_embd]` → `[batch, num_heads, seq_length, head_dim]`
- First reshape to add head dimension: `[batch, seq_length, num_heads, head_dim]`
- Then transpose to move heads: `[batch, num_heads, seq_length, head_dim]`
- GPT-2: 768 dims split into 12 heads × 64 dims/head
- Each head operates independently on its 64-dimensional subspace

**Head Merging**:
- Reverse of splitting: `[batch, num_heads, seq_length, head_dim]` → `[batch, seq_length, n_embd]`
- First transpose: `[batch, seq_length, num_heads, head_dim]`
- Then reshape to flatten heads: `[batch, seq_length, n_embd]`
- Concatenates all head outputs back into original dimension

**Parallel Attention**:
- Same attention mechanism as Step 08, but applied to all heads simultaneously
- Shape `[batch, num_heads, seq_length, head_dim]` allows broadcasting
- Q @ K^T operates on last two dimensions: `[seq_length, head_dim] @ [head_dim, seq_length]`
- All heads computed in single operation (highly efficient)

**Output Projection (c_proj)**:
- After merging heads, apply learned linear transformation
- Maps `[batch, seq_length, n_embd]` → `[batch, seq_length, n_embd]`
- Allows model to mix information across heads
- Essential for combining the different head perspectives

**HuggingFace Architecture**:
- `c_attn`: Combined Q/K/V projection to `3 * n_embd`
- `c_proj`: Output projection after merging heads
- This naming matches original GPT-2 implementation
- Required for loading pretrained weights

<div class="note">
<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Linear layers**:
- [`Linear(in_features, out_features, bias=True)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear): Q/K/V and output projections

**Tensor operations**:
- `tensor.reshape(new_shape)`: Splits or merges head dimension
- `tensor.transpose(axis1, axis2)`: Rearranges dimensions for parallel attention
- [`F.split(tensor, split_sizes, axis)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.split): Divides Q/K/V from combined projection

</div>

## Implementation tasks

1. **Import Required Modules** (Lines 13-15):
   - Copy imports from `solution_08.py` (math, F, Tensor, Device, DType, Dim, DimLike)
   - Add imports for `Linear` and `Module` from `max.nn.module_v3`
   - Copy the `causal_mask` function from Step 08

2. **Create Projection Layers** (Lines 38-43):
   - Create `c_attn`: `Linear(self.embed_dim, 3 * self.embed_dim, bias=True)`
   - Create `c_proj`: `Linear(self.embed_dim, self.embed_dim, bias=True)`
   - Store `self.num_heads` and `self.head_dim` from config

3. **Implement _split_heads** (Lines 58-64):
   - Calculate new shape: `tensor.shape[:-1] + [num_heads, attn_head_size]`
   - Reshape to add head dimension: `tensor.reshape(new_shape)`
   - Transpose to move heads to position 1: `tensor.transpose(-3, -2)`
   - Returns shape `[batch, num_heads, seq_length, head_size]`

4. **Implement _merge_heads** (Lines 79-85):
   - Transpose heads back: `tensor.transpose(-3, -2)`
   - Calculate flattened shape: `tensor.shape[:-2] + [num_heads * attn_head_size]`
   - Reshape to remove head dimension: `tensor.reshape(new_shape)`
   - Returns shape `[batch, seq_length, n_embd]`

5. **Implement _attn** (Lines 100-103):
   - Copy the attention computation from Step 08's `compute_attention` function
   - Same 5-step process: compute scores, scale, mask, softmax, weighted sum
   - Works identically but operates on all heads in parallel

6. **Implement Forward Pass** (Lines 117-138):
   - Project input: `qkv = self.c_attn(hidden_states)`
   - Split Q/K/V: `F.split(qkv, [self.split_size, self.split_size, self.split_size], axis=-1)`
   - Split heads for each: `self._split_heads(query, self.num_heads, self.head_dim)` (and for key, value)
   - Compute attention: `self._attn(query, key, value)`
   - Merge heads: `self._merge_heads(attn_output, self.num_heads, self.head_dim)`
   - Output projection: `self.c_proj(attn_output)`

**Implementation** (`step_09.py`):

```python
{{#include ../../steps/step_09.py}}
```

### Validation

Run `pixi run s09` to verify your implementation.

**Reference**: `solutions/solution_09.py`

**Next**: In [Step 10](./step_10.md), you'll implement residual connections and layer normalization to enable training deep transformer networks.
