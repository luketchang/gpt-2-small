import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTLanguageModelConfig:
  block_size: int = 1024,
  vocab_size: int = 50_257,
  n_embed: int = 768,
  n_heads: int = 12,
  n_blocks: int = 12,
  dropout_rate: int = 0.2,
  device: str = "cuda"

class Head(nn.Module):
    """A single self-attention head"""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embed // config.n_heads

        self.key = nn.Linear(config.n_embed, head_size, bias=False)
        self.query = nn.Linear(config.n_embed, head_size, bias=False)
        self.value = nn.Linear(config.n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)))
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        B, T, C = x.shape # batch size, block size, n_embed
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, config):
        super().__init__()

        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embed, config.n_embed) # NOTE: in the paper dims say n_heads * head_size, which is same as n_embed in our case
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # we concat on dim -1 because we want a (B, T, n_heads*head_size) tensor so we need to concat on the final dim, this ends up being same as (B, T, C) though because we configured head_size to be n_embed / num_heads
        proj = self.proj(out) # final projection, W^o in the paper
        return self.dropout(proj)

class FeedForward(nn.Module):
    """A simple feed-forward module"""

    def __init__(self, config):
        super().__init__()

        # the paper denotes "two linear transformations with a ReLU activation in between"
        # also note the 4x expansion in middle is due to detail in paper in the feedforward section
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout_rate)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer block"""

    def __init__(self, config):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(config.n_embed)
        self.sa = MultiHeadAttention(config)
        self.layernorm2 = nn.LayerNorm(config.n_embed)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        # add "x +" as part of residual connection (helps with vanishing gradients in deep network)
        # also note layernorm now comes before self-attention and feedforward despite paper saying after
        x_norm1 = self.layernorm1(x)
        x = x + self.sa(x_norm1)
        x_norm2 = self.layernorm2(x)
        x = x + self.ffwd(x_norm2)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTLanguageModelConfig):
        super().__init__()
        self.config = config

        # each token directly reads off the logits for the next token from a lookup table
        # NOTE: the embedding layer has vocab_size keys each of which has a n_embed dim value, nn.Embedding is basically just a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed) # token embeddings, takes (B,T) and outputs (B,T,C) where C is embedding size
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed) # position embeddings, outputs (B,T,C)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_blocks)]
        ) # list of transformer blocks
        self.layernorm = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size) # final linear layer, outputs (B,T,vocab_size)
        self.tie_weights() # share weights between token embedding and final linear layer (common practice)

    def tie_weights(self):
      self.lm_head.weight = self.token_embedding_table.weight

    # Takes input of shape (B,T) so B batches of T tokens (numbers)
    def forward(self, tokens):
        B, T = tokens.shape

        # tokens and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(tokens) # (B,T,C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.config.device)) # (T,C)
        x = token_embeddings + position_embeddings # broadcasting automatically turns position_embeddings into (B,T,C) by adding dim and repeating (T,C) B times

        # apply transformer blocks
        x = self.blocks(x)

        # apply layernorm
        x = self.layernorm(x)

        # pass self attention into final layer to convert to vocab size dims
        logits = self.lm_head(x) # (B,T,vocab_size)
        return logits

    def generate(self, tokens, max_new_tokens):
        # tokens is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop tokens to make sure it doesn't exceed block_size
            tokens_cropped = tokens[:, -self.config.block_size:]

            # get the predictions
            logits = self(tokens_cropped)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            tokens_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            tokens = torch.cat((tokens, tokens_next), dim=1) # (B, T+1)
        return tokens