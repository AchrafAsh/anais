import pandas as pd

from mysql.connector import connect, Error
from src.constants import db_config


def add_training_data(df: pd.DataFrame) -> None:
    '''
    Insert training data in the database (web app) from a dataframe

    Args:
        df (DataFrame) - labelled data where destination is the raw input and code is the world port index
    '''
    # check columns
    if not 'destination' in df.columns.tolist() or not 'code' in df.columns.tolist():
        raise Error('Data frame must have columns destination and code')

    cnx = connect(**db_config)
    cursor = cnx.cursor()

    for i in df.index.tolist():
        destination = df.loc[i, 'destination']
        code = df.loc[i, 'code']

        add_destination = ("INSERT INTO destinations "
                           "(destination, code) "
                           "VALUES (%s, %s)")

        cursor.execute(add_destination, (destination, code))
        # destination_id = cursor.lastrowid

    cnx.commit()
    cursor.close()
    cnx.close()


def get_training_data() -> pd.DataFrame:
    cnx = connect(**db_config)
    cursor = cnx.cursor()

    cursor.execute('SELECT destination, code FROM destinations')
    data = cursor.fetchall()

    cnx.commit()
    cursor.close()
    cnx.close()

    df = pd.DataFrame(columns=['destination', 'code'])
    i = 0
    for (destination, code) in data:
        df.loc[i] = [destination, code]
        i += 1
    return df


def add_prediction(destination: str, label: str) -> int:
    cnx = connect(**db_config)
    cursor = cnx.cursor()

    insert_prediction = ("INSERT INTO predictions "
                         "(destination, label) "
                         "VALUES (%s, %s)")

    cursor.execute(insert_prediction, (destination.upper(), label))
    prediction_id = cursor.lastrowid

    cnx.commit()
    cursor.close()
    cnx.close()
    return prediction_id


def rate_prediction(id: int, is_correct: int) -> None:
    cnx = connect(**db_config)
    cursor = cnx.cursor()

    update_prediction = ('UPDATE predictions '
                         'SET is_correct=%s '
                         'WHERE id=%s')

    cursor.execute(update_prediction, (is_correct, id))

    cnx.commit()
    cursor.close()
    cnx.close()
