import unittest

from distances import damerau_levenshtein_distance


class TestDistances(unittest.TestCase):
    def test_damerau_levenshtein_distance_returns_number(self):
        s1 = "BREST"
        s2 = "FRBES"
        self.assertIsInstance(damerau_levenshtein_distance(s1, s2), int)

    def test_damerau_levenshtein_distance_case_unsensitive(self):
        s1 = "BREST"
        s2 = "FRBES"
        self.assertEqual(
            damerau_levenshtein_distance(s1, s2),
            damerau_levenshtein_distance(s1, s2.lower())
        )
        self.assertEqual(
            damerau_levenshtein_distance(s1, s2),
            damerau_levenshtein_distance(s1.lower(), s2)
        )

    def test_damerau_levenshtein_distance_symetric(self):
        s1 = "BREST"
        s2 = "FRBES"
        self.assertEqual(
            damerau_levenshtein_distance(s1, s2),
            damerau_levenshtein_distance(s2, s1)
        )


if __name__ == "__main__":
    unittest.main()
