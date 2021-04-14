from typing import List, Dict


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
