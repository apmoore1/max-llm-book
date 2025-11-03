# Step 13: Language model head

<div class="note">
    Learn to add the final linear projection layer that converts hidden states to vocabulary logits for next-token prediction.
</div>

## Implementing the language model head

In this step you'll add the language model head to create the complete `GPT2LMHeadModel`. This final linear layer projects transformer outputs (768-dimensional) to vocabulary logits (50,257-dimensional).

For each position in the sequence, the model outputs a score for every possible next token. These logits are converted to probabilities using softmax, enabling next-token prediction.

The LM head bridges continuous representations to discrete vocabulary predictions. Without it, the transformer would output meaningless vectors. At 768 × 50,257 = 38.6M parameters, it represents about 33% of GPT-2's 117M total parameters. This makes it the single largest component in the model, even larger than all 12 transformer blocks combined.

## Understanding the language model head

**Linear Projection**:
- Maps `[batch, seq_length, n_embd]` → `[batch, seq_length, vocab_size]`
- Uses [`Linear(n_embd, vocab_size, bias=False)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear)
- Each position gets independent logits over vocabulary
- No activation function (raw logits are used)

**Logits vs Probabilities**:
- **Logits**: Raw scores before softmax, can be any real number
- **Probabilities**: After softmax, sum to 1, range [0, 1]
- For generation, we often work with logits directly (temperature, top-k, etc.)
- Softmax applied during generation, not in the model forward pass

**Vocabulary Projection**:
- Output dimension = `vocab_size = 50,257`
- Each position outputs 50,257 scores (one per vocabulary token)
- Higher score = model thinks that token is more likely next
- Shape: `[batch, seq_length, vocab_size]`

**No Bias Term**:
- `bias=False` means only weights, no bias vector
- Reduces parameters: saves 50,257 values (0.4% of model size)
- Common in language models (GPT-2, GPT-3, etc.)
- Bias provides little benefit for vocabulary prediction because layer normalization before the LM head already centers the activations. The bias would just add a constant to all logits, which doesn't change the relative probabilities after softmax.

**Complete Model Architecture**:
1. Input: Token IDs `[batch, seq_length]`
2. Embeddings: Token + position `[batch, seq_length, 768]`
3. Transformer blocks (×12): `[batch, seq_length, 768]` → `[batch, seq_length, 768]`
4. Final layer norm: `[batch, seq_length, 768]`
5. LM head: `[batch, seq_length, 768]` → `[batch, seq_length, 50257]`
6. Output: Logits `[batch, seq_length, 50257]`

<div class="note">
<div class="title">MAX operations</div>

You'll use the following MAX operations to complete this task:

**Linear layer**:
- [`Linear(in_features, out_features, bias=False)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear): Projects hidden states to vocabulary logits

</div>

## Implementation tasks

1. **Import Required Modules** (Lines 13-16):
   - Import `Linear` and `Module` from `max.nn.module_v3`
   - Import `GPT2Config` from `solutions.solution_01`
   - Import `GPT2Model` from `solutions.solution_12`

2. **Create Transformer** (Lines 32-33):
   - Initialize `self.transformer = GPT2Model(config)`
   - This is the complete transformer from Step 12

3. **Create LM Head** (Lines 36-38):
   - Initialize `self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)`
   - Projects hidden states to vocabulary logits
   - Note `bias=False`

4. **Implement Forward Pass** (Lines 51-59):
   - Get hidden states: `hidden_states = self.transformer(input_ids)`
   - Project to logits: `logits = self.lm_head(hidden_states)`
   - Return `logits`

**Implementation** (`step_13.py`):

```python
{{#include ../../steps/step_13.py}}
```

### Validation

Run `pixi run s13` to verify your implementation.

**Reference**: `solutions/solution_13.py`

**Next**: In [Step 14](./step_14.md), you'll implement text generation using sampling and temperature control to generate coherent text autoregressively.
