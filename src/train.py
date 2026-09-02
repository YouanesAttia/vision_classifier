import torch
from tqdm.auto import tqdm
from pathlib import Path
from model import save_model


def train_step(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             loss_fn: torch.nn.Module, 
             optimizer: torch.optim.Optimizer,
             device: torch.device):
    training_acc = 0
    training_loss = 0
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_logits = model(x)
        y_pred = torch.argmax(y_logits, dim=1)
        training_acc += (y_pred == y).sum().item()/len(y_pred)
        loss = loss_fn(y_logits, y)
        training_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    training_loss = training_loss / len(dataloader)
    training_acc = training_acc / len(dataloader)
    return training_loss, training_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    test_acc = 0
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_logits = model(x)
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()
            y_pred = torch.argmax(y_logits, dim=1)
            test_acc += (y_pred == y).sum().item()/len(y_pred)
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler: torch.optim.lr_scheduler = None):
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": []
    }
    max_acc = 0
    model.to(device)
    for epoch in tqdm(range(epochs)):
        current_lr = optimizer.param_groups[0]['lr']
        results["lr"].append(current_lr)
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        if scheduler:
            scheduler.step()

        print(
            f"Epoch: {epoch+1} | "
            f"LR: {current_lr:.6f} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if test_acc > max_acc:
            save_model(model, 'models', 'model.pth')
            max_acc = test_acc

    return results