# LyricGen

LyricGen is a Python script for generating song lyrics using a GPT-2 model.

## Requirements

- Python 3.6 or later
- PyTorch
- Transformers library from Hugging Face

## Note

- This model is trained on my own texts and lyrics. You are not allowed to use or modify any of the JSON files provided in this repository, as they are only presented as an example for how to create a dataset based on your own text material.
- This is a very early prototype and will be modofied drastically and dynamically soon.

## Usage

1. Import the necessary libraries and load the GPT-2 tokenizer:

```python
import re
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

2. Load the GPT-2 model with a custom configuration:

```python
config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False, output_attentions=False)
config.dropout = 0.1
config.weight_decay = 0.01

model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
```

3. Train the model on your dataset (like the JSON file provided)
