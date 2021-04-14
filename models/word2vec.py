import json
import nltk
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import gensim.downloader as api
import re
from tqdm import tqdm


class W2V:
    def __init__(self, classes):
        self.model = Word2Vec(size=300, window=5, min_count=1, workers=2)
        self.classes = classes

    @staticmethod
    def softmax(x):
        denominator = sum([np.exp(i) for i in x.values()])
        for key, value in x.items():
            x[key] = np.exp(value) / denominator
        return x

    def fit(self, sentences):
        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=self.model.corpus_count,
                         epochs=300, report_delay=1)

    def __call__(self, text):
        text = text.lower()
        text = re.split('; |, | \* | \n | > | / | + | _ | -', text)
        # remove unknown words
        known_words = []
        for word in text:
            if word in self.model.wv.vocab.keys():
                known_words.append(word)

        if len(known_words) == 0:
            sims = []
        else:
            sims = self.model.most_similar(
                known_words, topn=10)  # [(str, distance)]

        labels = {}
        for label in self.classes:
            labels[label] = 0
        for i in range(len(sims)):
            if sims[i][0].upper() in self.classes:
                labels[sims[i][0].upper()] += 1

        return max(labels.items(), key=lambda x: x[1])[0], self.softmax(labels)

    def eval(self, dataset):
        corrects = 0
        errors = []

        for i in tqdm(range(len(dataset))):
            text = dataset.iloc[i]["destination"]
            target = dataset.iloc[i]["code"]

            pred, _ = self(text)
            if pred == target:
                corrects += 1
            else:
                errors.append((text, target, pred))

        return corrects / len(dataset), errors
