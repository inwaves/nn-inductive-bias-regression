import numpy as np
import unittest
from maths import normalise_data


class TestNormaliseData(unittest.TestCase):
    def test_normalise_with_empty_test_set(self):
        x_train = np.array([1, 2, 3, 4, 5])
        x_test = np.array([])

        normalised_train, xmin, xmax = normalise_data(x_train)
        normalised_test, _, _ = normalise_data(x_test, xmin, xmax)
        expected_tr = (2*(x_train-xmin) / (xmax-xmin))-1
        print(expected_tr, normalised_train)
        self.assertTrue(np.all(normalised_train == expected_tr))
        self.assertEqual(type(normalised_test), np.ndarray)
        self.assertEqual(normalised_test.size, 0)

    def test_normalise_with_disjunct_test_set(self):
        x_train = np.array([1, 2, 3, 4, 5])
        x_test = np.array([6, 7, 8, 9, 10])

        normalised_train, xmin, xmax = normalise_data(x_train)
        normalised_test, _, _ = normalise_data(x_test, xmin, xmax)
        expected_tr = (2*(x_train-xmin) / (xmax-xmin))-1
        expected_te = (2*(x_test-xmin) / (xmax-xmin))-1
        self.assertTrue(np.all(normalised_train == expected_tr))
        self.assertTrue(np.all(normalised_test == expected_te))
