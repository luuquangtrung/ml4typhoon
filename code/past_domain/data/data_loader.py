from torch.utils.data import Dataset, DataLoader
import torch
from multiprocessing import cpu_count

def create_dataloader(train_dataset: Dataset, 
                      val_dataset: Dataset, 
                      test_dataset: Dataset,
                      shuffle = True,
                      num_workers = 32
                      ) -> DataLoader:

    train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=32, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            pin_memory=True if torch.cuda.is_available() else False
        )

    validation_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, validation_loader, test_loader

