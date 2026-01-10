"""
Step 07: Multi-head Attention

Implement multi-head attention that splits Q/K/V into multiple heads,
computes attention in parallel for each head, and merges the results.

Tasks:
1. Import required modules (math, F, Tensor, Linear, Module, etc.)
2. Create c_attn and c_proj linear layers
3. Implement _split_heads: reshape and transpose to add head dimension
4. Implement _merge_heads: transpose and reshape to remove head dimension
5. Implement _attn: compute attention for all heads in parallel
6. Implement forward pass: project -> split -> attend -> merge -> project

Run: pixi run s07
"""

# TODO: Import required modules
# Hint: You'll need math for scaling
import math
from max.experimental import functional as F
from max.experimental.tensor import Tensor, DType
from max.driver import Device
from max.graph import Dim, DimLike
from max.nn.module_v3 import Linear, Module
# Hint: You'll need functional as F from max.experimental
# Hint: You'll need Tensor, Device, DType from max.experimental.tensor and max.driver
# Hint: You'll need Dim, DimLike from max.graph
# Hint: You'll also need Linear and Module from max.nn.module_v3

from solutions.solution_01 import GPT2Config


@F.functional
def causal_mask(
    sequence_length: DimLike,
    num_tokens: DimLike,
    *,
    dtype: DType,
    device: Device,
):
    """Create a causal mask for autoregressive attention.

    Args:
        sequence_length: Length of the sequence.
        num_tokens: Number of tokens.
        dtype: Data type for the mask.
        device: Device to create the mask on.

    Returns:
        A causal mask tensor.
    """
    # Calculate total sequence length
    n = Dim(sequence_length) + num_tokens

    # 3: Create a constant tensor filled with negative infinity
    # TODO: Use Tensor.constant() with float("-inf"), dtype, and device parameters
    # https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.constant
    # Hint: This creates the base mask value that will block attention to future tokens
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)

    # 4: Broadcast the mask to the correct shape
    # TODO: Use F.broadcast_to() to expand mask to shape (sequence_length, n)
    # https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.broadcast_to
    # Hint: This creates a 2D attention mask matrix
    mask = F.broadcast_to(mask, shape=(sequence_length, n))

    # 5: Apply band_part to create the causal (lower triangular) structure and return the mask
    # TODO: Use F.band_part() with num_lower=None, num_upper=0, exclude=True
    # https://docs.modular.com/max/api/python/experimental/functional/#max.experimental.functional.band_part
    # Hint: This keeps only the lower triangle, allowing attention to past tokens only
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)


class GPT2MultiHeadAttention(Module):
    """Multi-head attention for GPT-2."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        # TODO: Create combined Q/K/V projection
        # Hint: Use Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)

        # TODO: Create output projection
        # Hint: Use Linear(self.embed_dim, self.embed_dim, bias=True)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor: Tensor, num_heads: int, attn_head_size: int):
        """Split the last dimension into (num_heads, head_size).

        Args:
            tensor: Input tensor, shape [batch, seq_length, n_embd]
            num_heads: Number of attention heads
            attn_head_size: Dimension of each head

        Returns:
            Tensor with shape [batch, num_heads, seq_length, head_size]
        """
        # TODO: Add head dimension
        # Hint: new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        # Hint: tensor = tensor.reshape(new_shape)
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        # TODO: Move heads dimension to position 1
        # Hint: return tensor.transpose(-3, -2)
        return tensor.transpose(-3, -2)

    def _merge_heads(self, tensor: Tensor, num_heads: int, attn_head_size: int):
        """Merge attention heads back to original shape.

        Args:
            tensor: Input tensor, shape [batch, num_heads, seq_length, head_size]
            num_heads: Number of attention heads
            attn_head_size: Dimension of each head

        Returns:
            Tensor with shape [batch, seq_length, n_embd]
        """
        # TODO: Move heads dimension back
        # Hint: tensor = tensor.transpose(-3, -2)
        tensor = tensor.transpose(-3, -2)

        # TODO: Flatten head dimensions
        # Hint: new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        # Hint: return tensor.reshape(new_shape)
        new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor

    def _attn(self, query: Tensor, key: Tensor, value: Tensor):
        """Compute attention for all heads in parallel.

        Args:
            query: Query tensor, shape [batch, num_heads, seq_length, head_size]
            key: Key tensor, shape [batch, num_heads, seq_length, head_size]
            value: Value tensor, shape [batch, num_heads, seq_length, head_size]

        Returns:
            Attention output, shape [batch, num_heads, seq_length, head_size]
        """
        # TODO: Implement attention computation
        # The same 5-step process: scores, scale, mask, softmax, weighted sum
        query_scores = query @ key.transpose(-1, -2)
        scaled_query_scores = query_scores / math.sqrt(self.head_dim)
        mask = causal_mask(query.shape[-2], 0, dtype=query.dtype, device=query.device)
        scaled_query_scores_masked = scaled_query_scores + mask
        attn_weights = F.softmax(scaled_query_scores_masked)
        attn_output = attn_weights @ value
        return attn_output

    def __call__(self, hidden_states: Tensor):
        """Apply multi-head attention.

        Args:
            hidden_states: Input tensor, shape [batch, seq_length, n_embd]

        Returns:
            Attention output, shape [batch, seq_length, n_embd]
        """
        # TODO: Project to Q, K, V
        # Hint: qkv = self.c_attn(hidden_states)
        # Hint: query, key, value = F.split(qkv, [self.split_size, self.split_size, self.split_size], axis=-1)
        qkv = self.c_attn(hidden_states)
        q, k, v = F.split(qkv, [self.split_size, self.split_size, self.split_size], axis=-1)
        query = self._split_heads(q, self.num_heads, self.head_dim)
        key = self._split_heads(k, self.num_heads, self.head_dim)
        value = self._split_heads(v, self.num_heads, self.head_dim)
        attn_output = self._attn(query, key, value)
        merged_attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(merged_attn_output)
        return attn_output
