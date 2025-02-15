import unittest
from src.main.assignment_2 import neville_interpolation, newton_forward_interpolation

class TestNevilleInterpolation(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
