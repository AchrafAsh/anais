import pandas as pd
import numpy as np


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

    return unique_df
