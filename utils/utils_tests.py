import numpy as np
import unittest
from maths import normalise_data


class TestNormaliseData(unittest.TestCase):
    def test_normalise_with_empty_test_set(self):
        x_train = np.array([1, 2, 3, 4, 5])
        x_test = np.array([])

        normalised_train, normalised_test = normalise_data(x_train, x_test)
        expected_tr = np.array([1, 2, 3, 4, 5]) / 5
        self.assertTrue(np.all(normalised_train == expected_tr))
        self.assertEqual(type(normalised_test), np.ndarray)
        self.assertEqual(normalised_test.size, 0)
