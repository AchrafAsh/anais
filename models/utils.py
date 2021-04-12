import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import re


def best_k(k, data_dict):
    """
    Args: 
        k (int): number of best elements to return
        data_dict (dict): a dictionary with number values

    Returns:
        (list): list of tuples (label, value) sorted by decreasing values
    """

    data_list = list(data_dict.items())
    data_list.sort(key=lambda x: x[1], reverse=True)

    return data_list[:k]


def recall_at_k(k: int, preds: List[Dict], targets: List[str]) -> float:
    """Compute the recall at k score given a set of predicted labels and the true ones.

    Args:
        k (int): 
        preds (dict): predicted labels
        targets (list): true labels

    Returns:
        (float): Percentage of correct predicted labels
    """

    assert len(preds) == len(targets)
    n = len(preds)
    corrects = 0

    for i in range(n):
        output = best_k(k, preds[i])
        if targets[i] in list(map(lambda x: x[0], output)): corrects += 1

    return corrects / n


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
    s1, s2 = s1.lower(), s2.lower()
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


def regexp_processing(destination):
    """ Removes noise from destination
    FR LPE>NL RTM should return NLRTM
    NL RTM/FR DON should return FRDON

    Args:
        destination (str): initial destination input

    Returns:
        str: processed destination to remove noisy characters
    """
    pattern = re.compile("[>]+")
    matches = pattern.findall(destination)

    if matches:
        return destination.split(matches[-1])[-1].strip()
    else:
        return destination
