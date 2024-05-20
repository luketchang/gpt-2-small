import torch
import torch.nn as nn
from torch.nn import functional as F
import torchinfo
from model import GPTLanguageModelConfig, GPTLanguageModel
import os
from tqdm.auto import tqdm
from datetime import datetime
from train import train_step, val_step, accuracy_fn
from data import load_dataloaders

# Load wikitext data
dataloaders = load_dataloaders()
train_dataloader = dataloaders["train"]
val_dataloader = dataloaders["val"]

# Hyperparameters
batch_size = 8 
block_size = 1024 # max context size
learning_rate = 5e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 768
n_heads = 12
n_blocks = 12
dropout = 0.2
vocab_size = 50_257 # tiktoken gpt2 tokenizer vocab size (tiktoken.encoding_for_model("gpt2").n_vocab)

# Model config
config = GPTLanguageModelConfig(
    block_size = block_size,
    vocab_size = vocab_size,
    n_embed = n_embed,
    n_heads = n_heads,
    n_blocks = n_blocks,
    dropout_rate = dropout,
    device = device
)

# Print model summary
model = GPTLanguageModel(config).to(device)
torchinfo.summary(model, input_size=(1, config.block_size), dtypes=[torch.long])

# Set debugging env vars
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Loss fn and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Run train loop for 10 epochs
epochs = 10
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")

    train_step(model=model,
              loss_fn=loss_fn,
              optimizer=optimizer,
              data_loader=train_dataloader,
              accuracy_fn=accuracy_fn,
              device=device)
    val_step(model=model,
              data_loader=val_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn)

# Save model params to disk
date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
torch.save(model.state_dict(), f"gpt-2-small-{date}.pth")