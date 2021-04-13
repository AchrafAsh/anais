import numpy as np
import pandas as pd
from distances import damerau_levenshtein_distance
from data import regexp_processing
from tqdm import tqdm
from collections import Counter


class KNN:
    """
    K-nearest neigbours model.
    Args:
        k (int): number of nearst neighbours to consider
        classes (list): list of output classes
    """

    def __init__(self, k, classes):
        self.k = k
        self.destinations = {}
        self.classes = {label: 0 for label in classes}

    def fit(self, dataset):
        for i in range(len(dataset)):
            self.destinations[dataset.iloc[i]
                              ["destination"]] = dataset.iloc[i]["code"]

    def eval(self, dataset):
        """Evaluates the model on dataset

        Args:
            dataset (DataFrame): needs to have columns "destination" and "code" (i.e label)

        Returns:
            accuracy (float): the percentage of correct labels
        """

        correct = 0
        false_preds = []
        for i in tqdm(range(len(dataset))):
            text = dataset.iloc[i]["destination"]
            label = dataset.iloc[i]["code"]
            pred, _ = self(text)
            if pred == label:
                correct += 1
            else:
                false_preds.append((text, pred, label))

        return correct / len(dataset), false_preds

    @staticmethod
    def max_distance(text, elements):
        """Find the farthest element and returns its index
        Args:
            text (str): text to consider
            elements (double<str>): (destination, label)

        Returns:
            int: index of the farthest element
        """
        max_idx = 0
        max_distance = damerau_levenshtein_distance(text, elements[0][0])
        for i in range(1, len(elements)):
            if damerau_levenshtein_distance(text, elements[i][0]) > max_distance:
                max_idx = i
                max_distance = damerau_levenshtein_distance(
                    text, elements[i][0])
        return max_idx

    @staticmethod
    def alt_softmax(counter):
        denominator = sum([np.exp(y)*y for y in counter.values()])
        for key in counter.keys():
            counter[key] = np.exp(counter[key])*counter[key] / denominator

        return counter

    def __call__(self, text):
        """Classify text with the most common label amongst k-nearest neighbours

        Args:
            text (str): text to label

        Returns:
            label(str): the predicted label
            classes (dict): list of possible labels with their probability
        """

        nearest_neighbours = []
        for destination, label in self.destinations.items():
            if len(nearest_neighbours) < self.k:
                nearest_neighbours.append((destination, label))
            else:
                worst_neighbour_idx = self.max_distance(
                    text, nearest_neighbours)
                nearest_neighbours[worst_neighbour_idx] = (destination, label)

        # find the best label
        label_counts = Counter([nn[1] for nn in nearest_neighbours])

        label_counts = self.alt_softmax(label_counts)
        for label in self.classes.keys():
            self.classes[label] = label_counts[label]

        return label_counts.most_common()[0][0], self.classes
