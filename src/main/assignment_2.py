# src/main/assignment_2.py
import numpy as np

# Perform Neville's method for polynomial interpolation.
def neville_interpolation(x_vals, y_vals, x_target):
    n = len(x_vals)
    p = np.zeros((n, n))

    for i in range(n):
        p[i][0] = y_vals[i]

    for j in range(1, n):
        for i in range(n - j):
            p[i][j] = ((x_target - x_vals[i + j]) * p[i][j - 1] - (x_target - x_vals[i]) * p[i + 1][j - 1]) / (x_vals[i] - x_vals[i + j])

    return p[0][-1]

# Perform Newton's divided difference method for polynomial interpolation.
def newton_forward_interpolation(x_vals, y_vals, degree, x_target=None):
    n = len(x_vals)
    table = np.zeros((n, n))
    table[:, 0] = y_vals
    
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_vals[i + j] - x_vals[i])

    coefficients = [table[0][j] for j in range(degree + 1)]
    
    if x_target is not None:
        approx = coefficients[0]
        term = 1
        for i in range(1, degree + 1):
            term *= (x_target - x_vals[i - 1])
            approx += coefficients[i] * term
        return coefficients, table, approx
    
    return coefficients, table, None

if __name__ == "__main__":
    # Neville's method example
    x = [3.6, 3.8, 3.9]
    y = [1.675, 1.436, 1.318]
    w = 3.7
    solution = neville_interpolation(x_vals=x, y_vals=y, x_target=w)
    print(f"Neville's method, Approximation of f(3.7): {solution:.7f}\n")
    
    # Newton's forward method example
    x_newton = [7.2, 7.4, 7.5, 7.6]
    y_newton = [23.5492, 25.3913, 26.8224, 27.4589]
    degree = 3
    x_target = 7.3
    coefficients, table, approx_f73 = newton_forward_interpolation(x_newton, y_newton, degree, x_target)
    
    print("Newton's Forward Coefficients:")
    for i in range(1,len(coefficients)):
        print(f"{coefficients[i]:.15f}")
    
    print(f"\nf(7.3) ≈ {approx_f73:.15f}")

    