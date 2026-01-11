"""
Step 08: Residual Connections and Layer Normalization

Implement layer normalization and residual connections, which enable
training deep transformer networks by stabilizing gradients.

Tasks:
1. Import F (functional), Tensor, DimLike, and Module
2. Create LayerNorm class with learnable weight and bias parameters
3. Implement layer norm using F.layer_norm
4. Implement residual connection (simple addition)

Run: pixi run s08
"""

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DimLike
from max.nn.module_v3 import Module


class LayerNorm(Module):
    """Layer normalization module matching HuggingFace GPT-2."""

    def __init__(self, dim: DimLike, *, eps: float = 1e-5):
        """Initialize layer normalization.

        Args:
            dim: Dimension to normalize (embedding dimension)
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = Tensor.ones([dim])
        self.bias = Tensor.zeros([dim])

    def __call__(self, x: Tensor) -> Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor, shape [..., dim]

        Returns:
            Normalized tensor, same shape as input
        """
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)


class ResidualBlock(Module):
    """Demonstrates residual connections with layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialize residual block.

        Args:
            dim: Dimension of the input/output
            eps: Epsilon for layer normalization
        """
        super().__init__()

        self.ln = LayerNorm(dim, eps=eps)
    def __call__(self, x: Tensor, sublayer_output: Tensor) -> Tensor:
        """Apply residual connection.

        Args:
            x: Input tensor (the residual)
            sublayer_output: Output from sublayer applied to ln(x)

        Returns:
            x + sublayer_output
        """
        return x + sublayer_output


def apply_residual_connection(input_tensor: Tensor, sublayer_output: Tensor) -> Tensor:
    """Apply a residual connection by adding input to sublayer output.

    Args:
        input_tensor: Original input (the residual)
        sublayer_output: Output from a sublayer (attention, MLP, etc.)

    Returns:
        input_tensor + sublayer_output
    """
    return input_tensor + sublayer_output
