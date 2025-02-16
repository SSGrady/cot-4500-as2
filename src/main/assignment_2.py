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

# Perform Hermite polynomial interpolation.
def hermite_interpolation(x_vals, f_vals, fprimes):
    n = len(x_vals)
    m = 2*n

    # z holds the repeated x-values, T holds the divided differences
    z = np.zeros(m)
    T = np.zeros((m, m))
    
    # Fill repeated x-values and f(x) in the 0th column of T
    for i in range(n):
        z[2*i]   = x_vals[i]
        z[2*i+1] = x_vals[i]
        T[2*i,   0] = f_vals[i]
        T[2*i+1, 0] = f_vals[i]
    
    # Fill the first-difference column (index 1 in T)
    for i in range(n):
        T[2*i,   1] = 0 if i == 0 else (f_vals[i] - f_vals[i-1]) / (x_vals[i] - x_vals[i-1])
        T[2*i+1, 1] = fprimes[i]
    
    # Fill higher-order columns j=2..(m-1)
    for j in range(2, m):
        for i in range(m - j):
            numerator = T[i+1, j-1] - T[i, j-1]
            denominator = z[i+j] - z[i]
            T[i, j] = numerator / denominator if abs(denominator) > 1e-15 else 0.0
    
    # Build the final matrix
    final = np.zeros((m, m))
    for i in range(m):
        final[i, 0] = z[i]        # x
        final[i, 1] = T[i, 0]     # f(x)
        # jth column of T is the jth column of final
        for j in range(1, m-1):
            if j < m:
                final[i, j+1] = T[i, j]
    
    return final

# Implement cubic spline interpolation.
def cubic_spline_system(x, f):
    n = len(x)
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Natural boundary conditions
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    
    h = np.diff(x)
    # Slope differences
    sd1 = (f[2] - f[1]) / h[1] - (f[1] - f[0]) / h[0]
    sd2 = (f[3] - f[2]) / h[2] - (f[2] - f[1]) / h[1]
    
    # Interior row for i=1
    A[1, 0] = h[0]
    A[1, 1] = 2*(h[0] + h[1])
    A[1, 2] = h[1]
    b[1] = 3*sd1  # factor of three
    
    # Interior row
    A[2, 1] = h[1]
    A[2, 2] = 2*(h[1] + h[2])
    A[2, 3] = h[2]
    b[2] = 3*sd2
    
    return A, b



if __name__ == "__main__":
    # Neville's method example
    x = [3.6, 3.8, 3.9]
    y = [1.675, 1.436, 1.318]
    w = 3.7
    solution = neville_interpolation(x_vals=x, y_vals=y, x_target=w)
    print(f"Neville's method, Approximation of f(3.7):\n{solution}\n")
    
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

    # Hermite polynomial interpolation example
    x = [3.6, 3.8, 3.9]
    f = [1.675, 1.436, 1.318]
    fp = [-1.195, -1.188, -1.182]
    table = hermite_interpolation(x, f, fp)

    print("\nHermite polynomial Interpolation matrix:")
    for row in table:
        print("[ " + " ".join(f"{val:.9e}" for val in row) + " ]")

    # Cubic spline interpolation example
    x_data = [2, 5, 8, 10]
    f_data = [3, 5, 7, 9]
    
    A, b = cubic_spline_system(x_data, f_data)
    x_sol = np.linalg.solve(A, b)
    
    print(f"\nCubic Spline Matrix:\n{A}\n{b}\n{x_sol}")