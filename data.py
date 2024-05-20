import torch
from torch.utils.data import Dataset
import tiktoken
from datasets import load_dataset


class WikitextDataset(Dataset):
    def __init__(self, dataset, block_size):
      super().__init__()
      self.block_size = block_size
      self.tokenizer = tiktoken.encoding_for_model("gpt2")

      data = [text for text in dataset['text'] if text != ""] # remove empty samples
      data = " ".join(data) # combine all samples into single string
      self.data = self.tokenizer.encode(data) # tokenize items in string

    def __len__(self):
      return ((len(self.data) - self.block_size) // self.block_size)

    def __getitem__(self, idx):
      start_idx = idx * self.block_size
      x = self.data[start_idx:start_idx+self.block_size]
      y = self.data[start_idx+1:start_idx+self.block_size+1]
      return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
def load_dataloaders(block_size=1024, batch_size=8):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    train_dataset = WikitextDataset(dataset['train'], block_size=block_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    val_dataset = WikitextDataset(dataset['validation'], block_size=block_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = WikitextDataset(dataset['test'], block_size=block_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
    }
