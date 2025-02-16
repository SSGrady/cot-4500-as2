# src/test/test_assignment_2.py
import unittest
import numpy as np
from src.main.assignment_2 import neville_interpolation, newton_forward_interpolation, hermite_interpolation, cubic_spline_system

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
        x = [3.6, 3.8, 3.9]
        f = [1.675, 1.436, 1.318]
        fp = [-1.195, -1.188, -1.182]
        # Hard coded expected matrix as per assignment instructions
        expected = np.array([
            [3.6,  1.675,  0.0,      -5.975,       29.875,      -98.80555556],
            [3.6,  1.675, -1.195,     0.0,          0.23333333,  -0.44444444],
            [3.8,  1.436, -1.195,     0.07,         0.1,         0.0],
            [3.8,  1.436, -1.188,     0.08,         0.0,         0.0],
            [3.9,  1.318, -1.18,      0.0,          0.0,         0.0],
            [3.9,  1.318, -1.182,     0.0,          0.0,         0.0]
        ])
        result = hermite_interpolation(x, f, fp)
        np.testing.assert_allclose(result, expected, rtol=1e-05, atol=1e-05)
    
    def test_cubic_spline_system(self):
        # Cubic spline interpolation example data
        x_data = [2, 5, 8, 10]
        f_data = [3, 5, 7, 9]
        # Hard coded expected matrix as per assignment instructions
        expected_A = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [3.0, 12.0, 3.0, 0.0],
            [0.0, 3.0, 10.0, 2.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        expected_b = np.array([0.0, 0.0, 1.0, 0.0])
        expected_x = np.array([0.0, -0.02702703, 0.10810811, 0.0])
        
        A, b = cubic_spline_system(x_data, f_data)
        x_sol = np.linalg.solve(A, b)
        
        np.testing.assert_allclose(A, expected_A, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(b, expected_b, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(x_sol, expected_x, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
