import string
import numpy as np
import pandas as pd
from models.utils import damerau_levenshtein_distance, regexp_processing
from tqdm import tqdm
from collections import Counter
from itertools import product


def create_ngram(n=2):
    combs = product(string.ascii_lowercase, repeat=n)
    ngrams = []
    for el in combs:
        ngrams.append("".join(el))

    return ngrams


class NaiveBayes:
    def __init__(self, n, classes):
        """
        Args:
            n (int)
        """
        self.ngrams = create_ngram(n)
        self.probs = pd.DataFrame(1, index=np.arange(
            len(self.ngrams)).tolist(), columns=classes)
        self.classes = classes
        self.y_prob = {}
        for label in classes:
            self.y_prob[label] = 0

    def fit(self, dataset):
        total_grams_per_class = {}
        for label in self.classes:
            total_grams_per_class[label] = 0

        for idx in tqdm(range(len(dataset))):
            text = dataset.iloc[idx]["destination"]
            target = dataset.iloc[idx]["code"]

            self.y_prob[target] += 1
            total_grams_per_class[target] += 1

            for idx in range(len(self.ngrams)):
                if self.ngrams[idx] in text.lower():
                    self.probs.loc[idx, target] += 1

        # divide by the total number of word for each column
        for target in self.classes:
            self.probs[target] /= total_grams_per_class[target] * \
                len(self.ngrams)
            self.y_prob[target] /= sum(self.y_prob.values())

    def predict(self, text):
        pred_targets = {}
        for label in self.classes:
            pred_targets[label] = 1

        for idx in range(len(self.ngrams)):
            if self.ngrams[idx] in text.lower():
                for label in self.classes:
                    pred_targets[label] *= self.y_prob[label] * \
                        self.probs.loc[idx, label]

        label = max(pred_targets, key=pred_targets.get)
        return label, pred_targets

    def eval(self, dataset):
        correct = 0
        errors = []

        for idx in tqdm(range(len(dataset))):
            text = dataset.iloc[idx]["destination"]
            target = dataset.iloc[idx]["code"]
            label, _ = model(text)
            if label == target:
                correct += 1
            else:
                errors.append((text, target, label))

        accuracy = correct / len(dataset)
        print(f"Accuracy: {accuracy} | Errors: {len(errors)}")
        return accuracy, errors

    def __call__(self, text): return self.predict(text)
