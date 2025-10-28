# Step 04: Implement GPT-2 MLP (Feed-Forward Network)

**Purpose**: Build the feed-forward network (MLP) component that processes information after attention in each transformer block.

## What is the GPT-2 MLP?

The Multi-Layer Perceptron (MLP) is a feed-forward neural network that appears in every transformer block of GPT-2, immediately after the attention mechanism. It consists of two linear transformations with a non-linear activation function in between.

In practice, the MLP serves as a position-wise feed-forward network that processes each token independently. It expands the representation to a larger intermediate dimension, applies a non-linear transformation, then projects back to the original embedding dimension.

This creates a two-layer neural network with a characteristic "expansion and contraction" pattern that helps the model learn complex transformations of the attention outputs.

## Why Use an MLP in Transformers?

**1. Non-Linear Transformations**: While attention provides a powerful mechanism for aggregating information across tokens, it's fundamentally a linear operation (weighted sum). The MLP adds crucial non-linearity through the GELU activation function, enabling the model to learn complex patterns.

**2. Position-Wise Processing**: The MLP processes each position independently (unlike attention which looks across positions), allowing the model to refine and transform the attended representations at each position.

**3. Capacity and Expressiveness**: The intermediate layer expansion (typically 4x the embedding dimension in GPT-2) provides additional capacity for the model to learn rich transformations. This expansion is critical for model performance.

**4. Information Mixing**: While attention mixes information across sequence positions, the MLP mixes information across feature dimensions at each position, providing a complementary form of computation.

### Key Concepts:

**MLP Architecture**:
- Two linear layers: `c_fc` (expansion) and `c_proj` (projection)
- `c_fc`: Projects from embedding dimension (768) to intermediate size (typically 3072 = 4�768)
- `c_proj`: Projects from intermediate size back to embedding dimension
- Non-linear activation (GELU) between the two layers
- Both layers use bias terms

**GELU Activation Function**:
- GELU (Gaussian Error Linear Unit) is the activation function used in GPT-2
- Smoother alternative to ReLU, incorporating probabilistic behavior
- Formula: `GELU(x) H x * �(x)` where � is the cumulative distribution function of the standard normal distribution
- The `approximate="tanh"` parameter uses a faster tanh-based approximation: `0.5 * x * (1 + tanh((2/�) * (x + 0.044715 * x�)))`
- Provides smooth gradients and better training dynamics than ReLU

**MAX Linear Layers**:
- [`Linear(in_features, out_features, bias=True)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear): Applies linear transformation `y = xW^T + b`
- `in_features`: Size of input feature dimension
- `out_features`: Size of output feature dimension
- `bias`: Whether to include a learnable bias term (GPT-2 uses bias=True)

**MAX GELU Function**:
- [`F.gelu(input, approximate="tanh")`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.gelu): Applies GELU activation
- `input`: Input tensor to transform
- `approximate`: Approximation method - "tanh" for faster computation (matches GPT-2), "none" for exact calculation

**Layer Naming Convention**:
- `c_fc`: "c" prefix is HuggingFace convention, "fc" stands for "fully connected" (the expansion layer)
- `c_proj`: "proj" stands for "projection" (projects back to embedding dimension)
- These names match the original GPT-2 checkpoint structure for weight loading compatibility

### Implementation Tasks (`step_04.py`):

1. **Import Required Modules** (Lines 1-9):
   - `functional as F` from `max.experimental` - provides F.gelu() activation function
   - `Tensor` from `max.experimental.tensor` - tensor operations (used implicitly)
   - `Linear` from `max.nn.module_v3` - linear transformation layers
   - `Module` from `max.nn.module_v3` - base class for neural network modules

2. **Create First Linear Layer (c_fc)** (Lines 25-29):
   - Use `Linear(embed_dim, intermediate_size, bias=True)`
   - This is the expansion layer that increases dimensionality
   - Maps from embedding dimension (768) to intermediate size (typically 3072)
   - Stores in `self.c_fc`

3. **Create Second Linear Layer (c_proj)** (Lines 31-35):
   - Use `Linear(intermediate_size, embed_dim, bias=True)`
   - This is the projection layer that restores original dimensionality
   - Maps from intermediate size back to embedding dimension
   - Stores in `self.c_proj`

4. **Apply First Linear Transformation** (Lines 46-49):
   - Apply `self.c_fc(hidden_states)` to expand the representation
   - This transforms shape from `[batch, seq_len, embed_dim]` to `[batch, seq_len, intermediate_size]`
   - Reassign result to `hidden_states`

5. **Apply GELU Activation** (Lines 51-55):
   - Use `F.gelu(hidden_states, approximate="tanh")`
   - Applies non-linear transformation element-wise
   - The `approximate="tanh"` matches GPT-2's implementation for efficiency
   - Reassign result to `hidden_states`

6. **Apply Second Linear Transformation** (Lines 57-60):
   - Apply `self.c_proj(hidden_states)` to project back to original dimension
   - This transforms shape from `[batch, seq_len, intermediate_size]` back to `[batch, seq_len, embed_dim]`
   - Return the final result

**Implementation**:
```python
# Import required modules from MAX
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear, Module

from solutions.solution_01 import GPT2Config


class GPT2MLP(Module):
    """Feed-forward network matching HuggingFace GPT-2 structure."""

    def __init__(self, intermediate_size: int, config: GPT2Config):
        super().__init__()
        embed_dim = config.n_embd

        # Create expansion layer
        self.c_fc = Linear(embed_dim, intermediate_size, bias=True)

        # Create projection layer
        self.c_proj = Linear(intermediate_size, embed_dim, bias=True)

    def __call__(self, hidden_states: Tensor) -> Tensor:
        """Apply feed-forward network."""
        # Expand to intermediate dimension
        hidden_states = self.c_fc(hidden_states)

        # Apply non-linear activation
        hidden_states = F.gelu(hidden_states, approximate="tanh")

        # Project back to embedding dimension
        hidden_states = self.c_proj(hidden_states)
        return hidden_states
```

### Validation:
Run `pixi run s04`

A failed test will show:
```bash
Running tests for Step 04: Implement GPT-2 MLP (Feed-Forward Network)...

Results:
❌ functional module is not imported from max.experimental
   Hint: Add 'from max.experimental import functional as F'
❌ Tensor is not imported from max.experimental.tensor
   Hint: Add 'from max.experimental.tensor import Tensor'
❌ Linear and Module are not imported from max.nn.module_v3
   Hint: Add 'from max.nn.module_v3 import Linear, Module'
❌ GPT2MLP class not found in step_04 module
❌ self.c_fc Linear layer is not created correctly
   Hint: Use Linear(embed_dim, intermediate_size, bias=True)
❌ self.c_proj Linear layer is not created correctly
   Hint: Use Linear(intermediate_size, embed_dim, bias=True)
❌ self.c_fc is not applied to hidden_states
   Hint: Apply self.c_fc to hidden_states in the __call__ method
❌ F.gelu is not used
   Hint: Use F.gelu() for the activation function
❌ self.c_proj is not applied to hidden_states
   Hint: Apply self.c_proj to hidden_states after the activation
❌ Found placeholder 'None' values that need to be replaced:
   self.c_fc = None
   self.c_proj = None
   hidden_states = None
   return None
   Hint: Replace all 'None' values with the actual implementation

============================================================
⚠️ Some checks failed. Review the hints above and try again.
============================================================
```

A successful test will show:
```bash
Running tests for Step 04: Implement GPT-2 MLP (Feed-Forward Network)...

Results:
✅ functional module is correctly imported as F from max.experimental
✅ Tensor is correctly imported from max.experimental.tensor
✅ Linear and Module are imported from max.nn.module_v3
✅ GPT2MLP class exists
✅ self.c_fc Linear layer is created correctly
✅ self.c_proj Linear layer is created correctly
✅ self.c_fc is applied to hidden_states
✅ F.gelu is used with approximate='tanh'
✅ self.c_proj is applied to hidden_states
✅ All placeholder 'None' values have been replaced
✅ GPT2MLP class can be instantiated
✅ GPT2MLP.c_fc is initialized
✅ GPT2MLP.c_proj is initialized
✅ GPT2MLP forward pass executes without errors
✅ Output shape is correct: (1, 4, 768)

============================================================
🎉 All checks passed! Your implementation matches the solution.
============================================================
```

**Reference**: `solutions/solution_04.py`
