# src/test/test_assignment_2.py
import unittest
import numpy as np
from src.main.assignment_2 import neville_interpolation, newton_forward_interpolation, hermite_interpolation

class TestAllInterpolationMethods(unittest.TestCase):
    def test_neville_interpolation(self):
        x = [3.6, 3.8, 3.9]
        y = [1.675, 1.436, 1.318]
        w = 3.7
        expected = 1.5549999999999995
        result = neville_interpolation(x_vals=x, y_vals=y, x_target=w)
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_newton_forward_interpolation(self):
        x = [7.2, 7.4, 7.5, 7.6]
        y = [23.5492, 25.3913, 26.8224, 27.4589]
        degree = 3
        x_target = 7.3
        expected_coeffs = [9.2105, 17.00166666666675, -141.82916666666722]
        expected_f73 = 24.016574999999992
        coefficients, _, approx_f73 = newton_forward_interpolation(x, y, degree, x_target)
        
        for i in range(1, len(expected_coeffs) + 1):
            self.assertAlmostEqual(coefficients[i], expected_coeffs[i - 1], places=5)
        self.assertAlmostEqual(approx_f73, expected_f73, places=5)
    
    def test_hermite_interpolation(self):
        # Inputs for Hermite interpolation
        x = [3.6, 3.8, 3.9]
        f = [1.675, 1.436, 1.318]
        fp = [-1.195, -1.188, -1.182]
        # Expected 6×6 matrix
        expected = np.array([
            [3.6,  1.675,  0.0,      -5.975,    29.875,   -98.80555556],
            [3.6,  1.675, -1.195,     0.0,       0.23333333, -0.44444444],
            [3.8,  1.436, -1.195,     0.07,      0.1,      0.0],
            [3.8,  1.436, -1.188,     0.08,      0.0,      0.0],
            [3.9,  1.318, -1.18,      0.0,       0.0,      0.0],
            [3.9,  1.318, -1.182,     0.0,       0.0,      0.0]
        ])
        result = hermite_interpolation(x, f, fp)
        # Compare each element to within 5 decimal places
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                self.assertAlmostEqual(result[i, j], expected[i, j], places=5)

if __name__ == "__main__":
    unittest.main()
