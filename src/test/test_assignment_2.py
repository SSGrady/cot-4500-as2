import unittest
from src.main.assignment_2 import neville_interpolation

class TestNevilleInterpolation(unittest.TestCase):
    def test_neville_interpolation(self):
        x = [3.6, 3.8, 3.9]
        y = [1.675, 1.436, 1.318]
        w = 3.7
        expected = 1.5549999999999995
        result = neville_interpolation(x_vals=x, y_vals=y, x_target=w)
        self.assertAlmostEqual(result, expected, places=5)

if __name__ == "__main__":
    unittest.main()
