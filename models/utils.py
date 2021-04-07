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


def damerau_levenshtein_distance(s1, s2):
    """Compute the Damerau-Levenshtein distance between two given
    strings (s1 and s2)

    Parameters:
        s1 (str): first string
        s2 (str): second string

    Returns:
        int: Return the damerau levenshtein distance (number of changes)
    """

    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1, lenstr1+1):
        d[(i, -1)] = i+1
    for j in range(-1, lenstr2+1):
        d[(-1, j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i-1, j)] + 1,  # deletion
                d[(i, j-1)] + 1,  # insertion
                d[(i-1, j-1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j-1] and s1[i-1] == s2[j]:
                # transposition
                d[(i, j)] = min(d[(i, j)], d[i-2, j-2] + cost)

    return d[lenstr1-1, lenstr2-1]
