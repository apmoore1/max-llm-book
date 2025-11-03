# Step 01: Model configuration

<div class="note">
    Learn to define the GPT-2 model architecture parameters using configuration classes.
</div>

## Defining the model architecture

Before you can implement GPT-2, you need to define its architecture - the dimensions, layer counts, and structural parameters that determine how the model processes information.

In this step, you'll create `GPT2Config`, a class that holds all the architectural decisions for GPT-2. This class describes things like: embedding dimensions, number of transformer layers, and number of attention heads. These parameters define the shape and capacity of your model.

OpenAI trained the original GPT-2 model with specific parameters that you can see in the [config.json file](https://huggingface.co/openai-community/gpt2/blob/main/config.json) on Hugging Face. By using the exact same values, we can later load OpenAI's pretrained weights.

## Understanding the parameters

The GPT-2 configuration consists of seven key parameters. Each one controls a different aspect of the model's architecture:

- `vocab_size`: Size of the token vocabulary (default: 50,257). This seemingly odd number is actually 50,000 BPE tokens + 256 byte-level tokens (fallback for rare characters) + 1 special token. GPT-2 uses byte-pair encoding (BPE) for tokenization.
- `n_positions`: Maximum sequence length, also called the context window (default: 1,024). This limit is a tradeoff between memory usage, computational cost, and the amount of context the model can attend to. Longer sequences require quadratic memory in attention.
- `n_embd`: Embedding dimension - the size of the hidden states that flow through the model (default: 768). This determines the model's capacity to represent information.
- `n_layer`: Number of transformer blocks stacked vertically (default: 12). More layers allow the model to learn more complex patterns.
- `n_head`: Number of attention heads per layer (default: 12). Multiple heads let the model attend to different types of patterns simultaneously.
- `n_inner`: Dimension of the MLP intermediate layer (default: 3,072). This is 4× the embedding dimension, a ratio found empirically in the original Transformer paper to work well.
- `layer_norm_epsilon`: Small constant for numerical stability in layer normalization (default: 1e-5). Prevents division by zero when variance is very small.

These values define the _small_ GPT-2 model. OpenAI released four sizes (small, medium, large, XL), each with different configurations that scale up these parameters.

## Implementing the configuration

Now let's implement this yourself. You'll create the `GPT2Config` class using Python's [`@dataclass`](https://docs.python.org/3/library/dataclasses.html) decorator. Dataclasses reduce boilerplate.

Instead of writing `__init__` and defining each parameter manually, you just declare the fields with type hints and default values.

First, you'll need to import the dataclass decorator from the dataclasses module. Then you'll add the `@dataclass` decorator to the `GPT2Config` class definition.

The actual parameter values come from Hugging Face. You can get them in two ways:

- **Option 1**: Run `pixi run huggingface` to access these parameters programmatically from the Hugging Face `transformers` library.
- **Option 2**: Read the values directly from the [GPT-2 model card](https://huggingface.co/openai-community/gpt2/blob/main/config.json).

Once you have the values, replace each `None` in the `GPT2Config` class properties with the correct numbers from the configuration.

**Implementation** (`step_01.py`):

```python
{{#include ../../steps/step_01.py}}
```

### Validation

Run `pixi run s01` to verify your implementation matches the expected configuration.

**Reference**: `steps/step_01.py`

**Next**: In [Step 02](./step_02.md), you'll implement causal masking to prevent tokens from attending to future positions in autoregressive generation.
