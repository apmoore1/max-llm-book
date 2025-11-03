# Step 11: Transformer block

<div class="note">
    Learn to combine attention, MLP, layer normalization, and residual connections into a complete transformer block.
</div>

## Implementing the transformer block

In this step you'll build the `Block` class that combines all previous components into a complete transformer block. This is the fundamental repeating unit that makes up GPT-2.

Each block has two sub-layers: multi-head attention (with layer norm and residual) and feed-forward MLP (with layer norm and residual). The pattern for each sub-layer is `x = x + sublayer(layer_norm(x))`.

GPT-2 stacks 12 identical transformer blocks. The pre-norm pattern (normalize before sublayer) is more stable than post-norm for deep networks. In pre-norm, gradients flow through both the normalized path and the residual path, preventing gradient explosion. Post-norm (`x = layer_norm(x + sublayer(x))`) can suffer from unstable gradients in very deep networks. Pre-norm enables effective training without careful initialization or learning rate warmup.

## Understanding the architecture

**Pre-norm Architecture**:
- Pattern: `output = input + sublayer(layer_norm(input))`
- Normalization happens first, before attention or MLP
- More stable than post-norm for deep networks
- Used in GPT-2, GPT-3, and most modern transformers

**Two Sub-layers**:
- **Attention sub-layer**: Multi-head self-attention from Step 09
- **MLP sub-layer**: Position-wise feed-forward from Step 04
- Each has its own layer norm and residual connection
- Process sequentially: attention first, then MLP

**Layer Dimensions**:
- `hidden_size = n_embd = 768` (embedding dimension)
- `inner_dim = 4 * hidden_size = 3072` (MLP inner dimension)
- `layer_norm_epsilon = 1e-5` (numerical stability)
- All sublayers maintain the 768-dimensional representation

**Information Flow**:
- Input: `[batch, seq_length, n_embd]`
- After attention block: same shape (residual preserves dimensions)
- After MLP block: same shape (residual preserves dimensions)
- Output: `[batch, seq_length, n_embd]`
- Dimension preservation allows stacking arbitrary numbers of blocks

**HuggingFace Naming**:
- `ln_1`: First layer norm (before attention)
- `attn`: Multi-head attention
- `ln_2`: Second layer norm (before MLP)
- `mlp`: Feed-forward network
- Matches original GPT-2 implementation for weight loading

## Implementation tasks

1. **Import Required Modules** (Lines 13-18):
   - Import `Module` from `max.nn.module_v3`
   - Import `GPT2Config` from `solutions.solution_01`
   - Import `GPT2MLP` from `solutions.solution_04`
   - Import `GPT2MultiHeadAttention` from `solutions.solution_09`
   - Import `LayerNorm` from `solutions.solution_10`

2. **Create Sub-layers** (Lines 40-53):
   - Create `ln_1`: `LayerNorm(hidden_size, eps=config.layer_norm_epsilon)`
   - Create `attn`: `GPT2MultiHeadAttention(config)`
   - Create `ln_2`: `LayerNorm(hidden_size, eps=config.layer_norm_epsilon)`
   - Create `mlp`: `GPT2MLP(inner_dim, config)`

3. **Implement Attention Block** (Lines 67-71):
   - Store residual: `residual = hidden_states`
   - Normalize: `hidden_states = self.ln_1(hidden_states)`
   - Apply attention: `attn_output = self.attn(hidden_states)`
   - Add residual: `hidden_states = attn_output + residual`

4. **Implement MLP Block** (Lines 74-78):
   - Store residual: `residual = hidden_states`
   - Normalize: `hidden_states = self.ln_2(hidden_states)`
   - Apply MLP: `feed_forward_hidden_states = self.mlp(hidden_states)`
   - Add residual: `hidden_states = residual + feed_forward_hidden_states`

5. **Return Output** (Line 81):
   - Return the final `hidden_states`

**Implementation** (`step_11.py`):

```python
{{#include ../../steps/step_11.py}}
```

### Validation

Run `pixi run s11` to verify your implementation.

**Reference**: `solutions/solution_11.py`

**Next**: In [Step 12](./step_12.md), you'll stack 12 transformer blocks together to create the complete GPT-2 model architecture.
