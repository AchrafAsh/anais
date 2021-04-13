import torch
from torch.utils.data import Dataset, DataLoader


class ANAISDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

        self.destinations = self.df["destination"]
        self.labels = self.df["code"]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        destination = self.destinations[idx]
        if self.transform:
            destination = self.transform(destination)
        return destination, self.labels[idx]


def get_loader(df, batch_size=10, num_workers=1, shuffle=True, pin_memory=True, transform=None):
    dataset = ANAISDataset(df, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=shuffle, pin_memory=pin_memory)
    return dataset, loader
