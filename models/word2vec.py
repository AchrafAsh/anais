import json, nltk, re
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import gensim.downloader as api
import nlpaug.augmenter.char as nac
from tqdm import tqdm

def word_augmentation(df, n):
    aug = nac.KeyboardAug()
    aug_df = pd.DataFrame(columns=['input','target','code'])
    
    for i in tqdm(range(len(df))):
        words = df.iloc[i]
        for j in range(n):
            augmented_data = aug.augment(words["input"])
            aug_df = aug_df.append({ "input": augmented_data, "target": words["target"], "code": words["code"] }, ignore_index=True)
    
    return df.append(aug_df)


def model(filename):
    df = pd.read_csv(filename)
    df['code'] = df['code'].apply(lambda code: code.replace(" ", ""))


    sentences = []
    for index, row in tqdm(df.iterrows()):
        words = nltk.word_tokenize(row['input']) + nltk.word_tokenize(row['target']) + nltk.word_tokenize(row['code'])
        words = [re.sub("[^A-Za-z']+", ' ', str(word)).lower() for word in words]
        sentences.append(words)

    # define the model
    w2v = Word2Vec(size = 300, window=5, min_count = 1, workers = 2)
    w2v.build_vocab(sentences)

    # train the model on the dataset
    w2v.train(sentences, total_examples=w2v.corpus_count,
            epochs=300, report_delay=1)
    
    return w2v


if __name__ == "__main__":
    model("")