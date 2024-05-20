# Reimplementation of GPT-2 (124M)

## model.py

Pytorch code containing `GPTLanguageModel` and `GPTLanguageModelConfig`. Follows the general specs below.

```
├── Token Embedding
├── Positional Embedding
├── Transformer Block (x12)
│   ├── LayerNorm
│   ├── MultiHeadAttention
│   ├── LayerNorm
│   └── FeedForward
├── LayerNorm
└── Linear (shares weights with token embedding table)
```

## data.py

Dataset implementation for HuggingFace Wikitext2 dataset + convience function for getting train/val/test dataloaders.

## train.py

Convenience functions for running train/validation steps and measuring accuracy.

## train_script.py

Train script for training GPT-2 124M. Loads data, sets up model, trains on train split and prints validation info every epoch. Below hyperparameters used as recorded for original GPT-2 paper.

| Hyperparameter               | Value |
| ---------------------------- | ----- |
| block size                   | 1024  |
| learning rate                | 5e-5  |
| embedding size               | 768   |
| number of attention heads    | 12    |
| number of transformer blocks | 12    |
| vocab size                   | 50257 |

NOTE: torchinfo summary says model has 163M parameters. This is because torchinfo does not pick up that final linear layer (39M params) shares weights with the token embedding table. Thus model actually has 163M - 39M = 124M params.
