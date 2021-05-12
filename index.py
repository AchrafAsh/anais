from app.db import add_prediction, add_training_data, get_training_data
from src.data import get_train_test_split


if __name__ == '__main__':
    # train, _ = get_train_test_split('10_ports.csv', split=1)
    # add_training_data(train)
    print(get_training_data())
