import unittest
import pandas as pd
from src.data import get_train_test_split


class TestDataLoaders(unittest.TestCase):
    def test_dataloader_returns_df(self):
        train, test = get_train_test_split()
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)

    def test_dataloader_seed_random(self):
        train, test = get_train_test_split()
        train_bis, test_bis = get_train_test_split()
        self.assertListEqual(train.index.tolist(), train_bis.index.tolist())
        self.assertListEqual(test.index.tolist(), test_bis.index.tolist())

    def test_dataloader_has_valid_columns(self):
        train, test = get_train_test_split()
        self.assertIn("code", train.columns)
        self.assertIn("destination", train.columns)
        self.assertIn("code", test.columns)
        self.assertIn("destination", test.columns)


if __name__ == "__main__":
    unittest.main()
