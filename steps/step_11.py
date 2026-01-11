"""
Step 11: Language Model Head

Add the final projection layer that converts hidden states to vocabulary logits.

Tasks:
1. Import Linear, Module, and previous components
2. Create transformer and lm_head layers
3. Implement forward pass: transformer -> lm_head

Run: pixi run s11
"""

from max.experimental.tensor import Tensor
from max.nn.module_v3 import Module, Linear
from solutions.solution_10 import GPT2Model
from solutions.solution_01 import GPT2Config


class MaxGPT2LMHeadModel(Module):
    """Complete GPT-2 model with language modeling head."""

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 with LM head.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, input_ids: Tensor):
        """Forward pass through transformer and LM head.

        Args:
            input_ids: Token IDs, shape [batch, seq_length]

        Returns:
            Logits over vocabulary, shape [batch, seq_length, vocab_size]
        """
        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
