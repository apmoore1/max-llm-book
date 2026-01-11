"""
Step 10: Stacking Transformer Blocks

Stack multiple transformer blocks with embeddings to create
the complete GPT-2 model architecture.

Tasks:
1. Import Tensor, Embedding, Module, Sequential, and previous components
2. Create token and position embeddings
3. Stack n_layer transformer blocks using Sequential
4. Create final layer normalization
5. Implement forward pass: embeddings -> blocks -> layer norm

Run: pixi run s10
"""

# TODO: Import required modules
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Module, Embedding, Sequential
from solutions.solution_01 import GPT2Config
from solutions.solution_08 import LayerNorm
from solutions.solution_09 import GPT2Block

class GPT2Model(Module):
    """Complete GPT-2 transformer model."""

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 model.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        # TODO: Create token embeddings
        # Hint: Use Embedding(config.vocab_size, dim=config.n_embd)
        self.wte = Embedding(config.vocab_size, dim=config.n_embd)

        # TODO: Create position embeddings
        # Hint: Use Embedding(config.n_positions, dim=config.n_embd)
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)

        # TODO: Stack transformer blocks
        # Hint: Use Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        # This creates config.n_layer blocks (12 for GPT-2 base)
        self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))

        # TODO: Create final layer normalization
        # Hint: Use LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(self, input_ids: Tensor):
        """Forward pass through the transformer.

        Args:
            input_ids: Token IDs, shape [batch, seq_length]

        Returns:
            Hidden states, shape [batch, seq_length, n_embd]
        """
        batch_size, sequence_length = input_ids.shape

        # TODO: Get token embeddings
        # Hint: tok_embeds = self.wte(input_ids)
        tok_embeds = self.wte(input_ids)
        pos_embeds = self.wpe(Tensor.arange(sequence_length,
                                                              dtype=input_ids.dtype,
                                                              device=input_ids.device))

        token_position_embeddings = tok_embeds + pos_embeds

        transformer_output = self.h(token_position_embeddings)
        norm_transformer_output = self.ln_f(transformer_output)
        return norm_transformer_output
