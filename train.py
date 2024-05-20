import torch
import torch.nn as nn

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    accuracy_fn,
    device: torch.device):
  model.train()

  train_acc, train_loss = 0, 0

  # train_dataloader iters (x, y), enumerate adds iter number (batch num)
  for batch_num, (X, y) in enumerate(data_loader):
    try:
      X, y = X.to(device), y.to(device)

      logits = model(X)
      B, T, C = logits.shape

      y_pred = logits.view(B*T, C)
      y = y.view(B*T)

      # add loss for every batch
      loss = loss_fn(y_pred, y)
      train_loss += loss
      train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=-1))

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

      if batch_num % 50 == 0:
        print(f"{batch_num * len(X)}/{len(data_loader.dataset)} samples...")
        curr_acc = train_acc / (batch_num + 1)
        curr_loss = train_loss / (batch_num + 1)
        print(f"Batch number {batch_num}. Train loss: {curr_loss:.4f} | Train acc: {curr_acc:.4f}%")
    except Exception as e:
        torch.set_printoptions(threshold=float('inf'))
        print("Exception:", e)
        print("X:", X)
        print("y:", y)
        raise e

  # get average loss per batch?
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)

  print(f"\nEnd of epoch. Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}%")

def val_step(model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              accuracy_fn,
              device: torch.device):
  """Performs a validation loop step on model going over data_loader."""
  val_loss, val_acc = 0, 0

  # Put the model in eval mode
  model.eval()

  # Turn on inference mode context manager
  with torch.inference_mode():
    for X, y in data_loader:
      # Send the data to the target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass (outputs raw logits)
      logits = model(X)
      B, T, C = logits.shape

      val_pred = logits.view(B*T, C)
      y = y.view(B*T)

      # 2. Calculuate the loss/acc
      val_loss += loss_fn(val_pred, y)
      val_acc += accuracy_fn(y_true=y,
                              y_pred=val_pred.argmax(dim=-1)) # go from logits -> prediction labels

    # Adjust metrics and print out
    val_loss /= len(data_loader)
    val_acc /= len(data_loader)
    print(f"Validation loss: {val_loss:.5f} | Validation acc: {val_acc:.2f}%\n")