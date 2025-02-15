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

if __name__ == "__main__":
    x = [3.6, 3.8, 3.9]
    y = [1.675, 1.436, 1.318]
    w= 3.7
    solution = neville_interpolation(x_vals=x, y_vals=y, x_target=w)
    print(f"Neville's method, Approximation of f(3.7):\n{solution}\n")