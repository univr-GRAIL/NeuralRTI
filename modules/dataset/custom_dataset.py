import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class Mycustomdataset(Dataset):  # Change to Dataset, not LightningModule
    _should_prevent_trainer_and_dataloaders_deepcopy = True  # Add this attribute

    def __init__(self, csv_file, info, data, transform=None, batch_size=32, num_workers=4):
        self.annotations = csv_file
        self.info = info
        self.transform = transform
        self.batch_size = batch_size  # Initialize batch_size
        self.num_workers = num_workers  # Initialize num_workers
        self.train_ds = self  # Assuming the dataset is the training set itself
        self.val_ds = self  # Same for validation (if applicable)
        self.all_pixels, self.all_lds = data
        self.p_idx = self.annotations[:, 0]
        self.gt_idx = self.annotations[:, 1]
        self.hw, self.no_lights, self.no_channel = self.info

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if self.transform:
            self.transform(torch.from_numpy(np.reshape(self.all_pixels[self.p_idx[index]])))

        return torch.from_numpy(np.reshape(self.all_pixels[self.p_idx[index]],
                                           (-1, self.no_lights * self.no_channel)).copy()), torch.from_numpy(
            self.all_lds[self.gt_idx[index]].copy()), torch.from_numpy(
            self.all_pixels[self.p_idx[index], self.gt_idx[index], 0:3].copy())

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,  # Typically you want shuffle for training data
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # Typically no shuffle for validation
        )
