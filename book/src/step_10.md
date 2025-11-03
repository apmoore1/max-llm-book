# Step 10: Residual connections and layer normalization

<div class="note">
    Learn to implement residual connections and layer normalization to enable training deep transformer networks.
</div>

## Implementing residual connections and layer normalization

In this step you'll combine residual connections and layer normalization into a reusable pattern for transformer blocks. Residual connections add the input directly to the output (`output = input + layer(input)`), creating shortcuts that help gradients flow through deep networks. Layer normalization stabilizes activations by normalizing across features for each position independently.

GPT-2 uses pre-norm architecture: layer norm is applied before each sublayer (attention or MLP), following the pattern `x = x + sublayer(layer_norm(x))`. This is more stable than post-norm for deep networks.

Residual connections create direct gradient paths through the network. During backpropagation, gradients can flow through the identity path (`x = x + ...`) without being multiplied by layer weights, preventing vanishing gradients in deep models. This allows training networks with 12+ layers. Without residuals, gradients would diminish exponentially as they propagate through many layers. Layer normalization stabilizes training by keeping activation distributions consistent, and works identically during training and inference because it normalizes each example independently.

## Understanding the components

**Layer Normalization Formula**:

$$\text{output} = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where:
- $\mu = \text{mean}(x)$ is the mean across the last dimension
- $\sigma^2 = \text{variance}(x)$ is the variance across the last dimension
- $\gamma$ is the learnable scale parameter (weight)
- $\beta$ is the learnable shift parameter (bias)
- $\epsilon$ prevents division by zero (typically 1e-5)

**MAX Layer Norm Implementation**:
- [`F.layer_norm(x, gamma, beta, epsilon)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.layer_norm)
- `gamma`: learnable scale parameter (initialized to 1)
- `beta`: learnable shift parameter (initialized to 0)
- Normalizes over the last dimension automatically

**Learnable Parameters**:
- `weight` (gamma): `Tensor.ones([dim])` - initialized to 1
- `bias` (beta): `Tensor.zeros([dim])` - initialized to 0
- These allow the network to learn optimal scaling and shifting

**Pre-norm Architecture**:
- GPT-2 uses the pre-norm pattern for residual connections:

$$\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

- Apply layer norm first, then sublayer, then add residual
- More stable than post-norm: $\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$

**Residual Addition**:
- Simple element-wise addition: `input + sublayer_output`
- Both tensors must have identical shapes
- No additional parameters needed (just addition)

<div class="note">
<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Layer normalization**:
- [`F.layer_norm(x, gamma, beta, epsilon)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.layer_norm): Normalizes across feature dimension

**Tensor initialization**:
- [`Tensor.ones([dim])`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.ones): Creates weight parameter
- [`Tensor.zeros([dim])`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.zeros): Creates bias parameter

</div>

## Implementation tasks

1. **Import Required Modules** (Lines 13-17):
   - Import `functional as F` from `max.experimental`
   - Import `Tensor` from `max.experimental.tensor`
   - Import `DimLike` from `max.graph`
   - Import `Module` from `max.nn.module_v3`

2. **Initialize LayerNorm Parameters** (Lines 33-38):
   - Create `self.weight`: `Tensor.ones([dim])`
   - Create `self.bias`: `Tensor.zeros([dim])`
   - Store `self.eps` for numerical stability

3. **Implement LayerNorm Forward Pass** (Lines 50-51):
   - Call `F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)`
   - Returns normalized tensor with same shape as input

4. **Create ResidualBlock LayerNorm** (Lines 68-69):
   - Initialize `self.ln = LayerNorm(dim, eps=eps)`
   - This will be used to normalize before sublayers

5. **Implement Residual Connection** (Lines 83-84):
   - Return `x + sublayer_output`
   - Simple addition creates the residual connection

6. **Implement apply_residual_connection** (Lines 97-98):
   - Return `input_tensor + sublayer_output`
   - Standalone function demonstrating the pattern

**Implementation** (`step_10.py`):

```python
{{#include ../../steps/step_10.py}}
```

### Validation

Run `pixi run s10` to verify your implementation.

**Reference**: `solutions/solution_10.py`

**Next**: In [Step 11](./step_11.md), you'll combine multi-head attention, MLP, layer norm, and residual connections into a complete transformer block.
