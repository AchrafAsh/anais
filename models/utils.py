import torch
from torch.utils.data import Dataset, DataLoader


def recall_at_k(k, preds, labels):
    """Compute the recall at k score given a set of predicted labels and the true ones.

    Parameters:
        k (int): 
        preds (tensor): predicted labels
        labels (tensor): true labels

    Returns:
        (float): Percentage of correct predicted labels
    """

    assert len(preds) == len(labels)
    n = len(preds)
    corrects = 0

    for i in range(n):
        if labels[i] in torch.topk(preds[i], k=k):
            corrects += 1

    return corrects / n


class ANAISDataset(Dataset):
    def __init__(self, df):
        self.df = df

        self.destinations = self.df["destination"]
        self.labels = self.df["code"]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        return self.destinations[idx], self.labels[idx]


def get_loader(df, batch_size=10, num_workers=1, shuffle=True, pin_memory=True):
    dataset = ANAISDataset(df)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=shuffle, pin_memory=pin_memory)
    return dataset, loader
