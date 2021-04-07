import nltk
import re
import nlpaug.augmenter.char as nac
import pandas as pd


def word_augmentation(df, n):
    aug = nac.KeyboardAug()
    aug_df = pd.DataFrame(columns=['input', 'target', 'code'])

    for i in range(len(df)):
        words = df.iloc[i]
        for j in range(n):
            augmented_data = aug.augment(words["input"])
            aug_df = aug_df.append(
                {"input": augmented_data, "target": words["target"], "code": words["code"]}, ignore_index=True)

    return df.append(aug_df)


def get_sentences(filename):
    df = pd.read_csv(filename)
    df['code'] = df['code'].apply(lambda code: code.replace(" ", ""))
    sentences = []
    for _, row in df.iterrows():
        words = nltk.word_tokenize(
            row['input']) + nltk.word_tokenize(row['target']) + nltk.word_tokenize(row['code'])
        words = [re.sub("[^A-Za-z']+", ' ', str(word)).lower()
                 for word in words]
        sentences.append(words)
    return sentences


def get_train_test_split(filename):
    df = pd.read_csv(filename)
    df['code'] = df['code'].apply(lambda code: code.replace(" ", ""))

    # shuffle
    df.sample(frac=1).reset_index(drop=True)

    # split train / test
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    return df_train, df_test