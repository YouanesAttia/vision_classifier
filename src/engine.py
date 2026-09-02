import torch
import data_setup, model, train
from pathlib import Path
from timeit import default_timer as timer 

def main():
    start_time = timer() 
    data_dir = Path('data')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataloader, test_dataloader, classes = data_setup.create_dataloaders(data_dir, batch_size=32)
    model0 = model.create_model(num_classes=2, freeze_base=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model0.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    results = train.train_model(model=model0, 
                                train_dataloader=train_dataloader, 
                                test_dataloader=test_dataloader,
                                loss_fn=loss_fn, 
                                optimizer=optimizer,
                                device=device, 
                                epochs=5,
                                scheduler=scheduler,
                                best_acc=0.9277)

    end_time = timer()
    total_time = end_time - start_time
    print(f"\nTotal training time: {total_time:.3f} seconds")
    
    if total_time > 60:
        print(f"Total training time: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()