import nltk
import re
import nlpaug.augmenter.char as nac
import numpy as np
import pandas as pd
from src.constants import constants


def get_full_df(filename="./data/ais_data/nari_static.csv"):
    types = {
        "sourcemmsi": "Int64",
        "imo": "Int64",
        "callsign": "str",
        "shipname": "str",
        "shiptype": "Int64",
        "to_bow": "Int64",
        "to_stern": "Int64",
        "to_starboard": "Int64",
        "to_port": "Int64",
        "eta": "str",  # estimated time of arrival
        "draught": np.float64,
        "destination": "str",
        "t": "Int64"
    }
    dataset = pd.read_csv(filename, dtype=types)

    # remove trailing spaces
    dataset["destination"] = dataset.apply(
        lambda row: str(row["destination"]).strip(), axis=1)

    # remove lines with no destination
    no_destination_rows = dataset[
        (dataset['destination'].isnull()) | (
            dataset['destination'] == "") | (dataset["destination"] == "nan")
    ].index.tolist()
    dataset = dataset.drop(no_destination_rows)

    # keep only unique values
    unique_df = dataset.loc[dataset["destination"].drop_duplicates(
    ).index.tolist()]
    unique_df = unique_df[["destination"]]


def word_augmentation(df, n):
    """Perform data augmentation with KeyboardAug, by inserting random keyboard inputs

    Parameters:
        df (DataFrame): dataset
        n (int): number of insertions for each row -> change to probability to insert one

    Returns:
        (DataFrame): the updated dataset with new rows
    """
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
    """
    Parameters:
        filename (str): path to the dataset (csv)

    Returns:
        (list): list of all rows with concatenated "{destination} {target} {code}"
    """
    df = pd.read_csv(filename)
    df['code'] = df['code'].apply(lambda code: code.replace(" ", ""))
    sentences = []
    for _, row in df.iterrows():
        words = nltk.word_tokenize(
            row['destination']) + nltk.word_tokenize(row['target']) + nltk.word_tokenize(row['code'])
        words = [re.sub("[^A-Za-z']+", ' ', str(word)).lower()
                 for word in words]
        sentences.append(words)
    return sentences


def get_train_test_split(filename="10_ports.csv", transform=None):
    """Split the dataset in train test data

    Parameters:
        filename (str): path to the data (csv)

    Returns:
        (df_train (DataFrame), df_test (DataFrame)): train and test datasets
    """
    df = pd.read_csv(filename)

    if transform:
        df["destination"] = df.apply(
            lambda x: transform(x["destination"]), axis=1)

    # shuffle
    df = df.sample(frac=1, random_state=constants["SEED"])

    # split train / test
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    return df_train, df_test


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
