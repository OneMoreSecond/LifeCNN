import unittest
from fractions import Fraction

import numpy as np

import model
import data
import utils

class TestMinimalModel(unittest.TestCase):

    def test_minimal_model(self):
        minimal_model = model.MinimalModel(initialization='ground_truth')
        test_data = data.MinimalDataset()
        accuracy = utils.eval(minimal_model, test_data)
        self.assertEqual(accuracy, Fraction(1))

if __name__ == '__main__':
    unittest.main()
