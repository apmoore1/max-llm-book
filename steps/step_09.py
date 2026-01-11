"""
Step 09: Transformer Block

Combine multi-head attention, MLP, layer normalization, and residual
connections into a complete transformer block.

Tasks:
1. Import Module and all previous solution components
2. Create ln_1, attn, ln_2, and mlp layers
3. Implement forward pass with pre-norm residual pattern

Run: pixi run s09
"""

# TODO: Import required modules
from mpmath import residual
from solutions.solution_01 import GPT2Config
from max.nn.module_v3 import Module
from solutions.solution_04 import GPT2MLP
from solutions.solution_07 import GPT2MultiHeadAttention
from solutions.solution_08 import LayerNorm


class GPT2Block(Module):
    """Complete GPT-2 transformer block."""

    def __init__(self, config: GPT2Config):
        """Initialize transformer block.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        hidden_size = config.n_embd
        inner_dim = config.n_inner

        # TODO: Create first layer norm (before attention)
        # Hint: Use LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # TODO: Create multi-head attention
        # Hint: Use GPT2MultiHeadAttention(config)
        self.attn = GPT2MultiHeadAttention(config)

        # TODO: Create second layer norm (before MLP)
        # Hint: Use LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # TODO: Create MLP
        # Hint: Use GPT2MLP(inner_dim, config)
        self.mlp = GPT2MLP(inner_dim, config)

    def __call__(self, hidden_states):
        """Apply transformer block.

        Args:
            hidden_states: Input tensor, shape [batch, seq_length, n_embd]

        Returns:
            Output tensor, shape [batch, seq_length, n_embd]
        """
        residual = hidden_states
        norm_hidden_state = self.ln_1(hidden_states)
        attn_output = self.attn(norm_hidden_state)
        attn_residual = attn_output + residual
        norm_attn_residual = self.ln_2(attn_residual)
        mlp_output = self.mlp(norm_attn_residual)
        output = mlp_output + attn_residual
        return output
