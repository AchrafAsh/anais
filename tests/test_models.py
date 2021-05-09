import unittest
import pandas as pd
from src.data import get_train_test_split
from src.constants import constants
from models.knn import KNN
from models.naive_bayes import NaiveBayes
from models.linear import Linear
from models.word2vec import W2V


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.train_df, self.test_df = get_train_test_split()
        self.classes = constants["classes"]

        self.KNN = KNN(k=4, classes=self.classes)
        self.KNN.fit(self.train_df)

        self.NaiveBayes = NaiveBayes(n=3, classes=self.classes)
        self.NaiveBayes.fit(self.train_df)

        self.Linear = Linear(classes=self.classes, max_len=40)
        self.Linear.fit(self.train_df, epochs=1)

        self.W2V = W2V(classes=self.classes)

    def test_knn_io(self):
        """
        Test that KNN model takes the right inputs and outputs a dictionary with all possible class
        """
        pred, output = self.KNN("BREST")
        self.assertIsInstance(output, dict)
        self.assertIn(pred, self.classes)
        for label in self.classes:
            self.assertIn(label, output.keys())

    def test_knn_output_probabilities(self):
        """
        Test that KNN model returns probabilities for each possible class
        """
        _, output = self.KNN("RADE DE BREST")
        # sums up to one
        self.assertLess(abs(sum(output.values()) - 1), 1e-3)
        # all values between 0 and 1
        for value in output.values():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)

    def test_knn_case_unsensitive(self):
        pred_upper, output_upper = self.KNN("BREST")
        pred_lower, output_lower = self.KNN("brest")

        self.assertEqual(pred_upper, pred_lower)
        self.assertListEqual(list(output_upper.items()),
                             list(output_lower.items()))

    def test_naive_bayes_io(self):
        """
        Test that Naive Bayes model takes the right inputs and outputs a dictionary with all possible class
        """
        pred, output = self.NaiveBayes("BREST")
        self.assertIn(pred, self.classes)
        self.assertIsInstance(output, dict)

    # def test_naive_bayes_output_probabilities(self):
    #     _, output = self.NaiveBayes("BREST")
    #     self.assertLess(abs(sum(output.values()) - 1), 1e-3)
    #     for label in self.classes:
    #         self.assertIn(label, output.keys())

    def test_linear_io(self):
        """
        Test that Linear model takes the right inputs and outputs a dictionary with all possible class
        """
        pred, output = self.Linear("BREST")
        self.assertIn(pred, self.classes)
        self.assertIsInstance(output, dict)

    def test_linear_output_probabilities(self):
        _, output = self.Linear("BREST")
        self.assertLess(abs(sum(output.values()) - 1), 1e-3)
        for label in self.classes:
            self.assertIn(label, output.keys())

    def test_w2v_io(self):
        """
        Test that Word2Vec model takes the right inputs and outputs a dictionary with all possible class
        """
        pred, output = self.W2V("BREST")
        self.assertIn(pred, self.classes)
        self.assertIsInstance(output, dict)

    def test_w2v_output_probabilities(self):
        _, output = self.W2V("BREST")
        self.assertLess(abs(sum(output.values()) - 1), 1e-3)
        for label in self.classes:
            self.assertIn(label, output.keys())


if __name__ == "__main__":
    unittest.main()
