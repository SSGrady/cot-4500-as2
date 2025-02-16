# COT4500 - Assignment 2

### Author: Steven Grady

## Overview

In this assignment, I implement numerical calculus methods for:

- **Neville's method**: for polynomial interpolation
- **Newton's forward method**: which builds a divided difference table and computes polynomial coefficients
- **Hermite polynomial interpolation**: using the divided difference method
- **Cubic spline interpolation**: where the system is built to find matrix A, vector b, and vector x.


## Requirements

- Python 3.x
- NumPy

## Files

- `src/main/assignment_2.py`: Main script for the assignment.

- `src/test/test_assignment_2.py`: Test script for all unit tests for the assignment.

## Run Instructions

1. Clone the repository using SSH and change to the root directory
```
git clone git@github.com:SSGrady/cot-4500-as2.git
cd cot-4500-as2
```

2. Run the main script:
```
python3 src/main/assignment_2.py
```

3. Run the test suite:
```
python3 -m unittest src.test.test_assignment_2
```

Final Output:

!["final output"](https://i.ibb.co/4nrs0pwr/Screenshot-2025-02-16-at-4-34-34-PM.png)
