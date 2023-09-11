import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def klee_minty_problem(n):
    # Define the objective function coefficients
    c = np.array([-2**(n - j) for j in range(n)])
    
    # Define the constraint matrix A and bounds
    A = np.zeros((n, n))
    for i in range(n):
        A[i, :i+1] = [2**(i - j) for j in range(i+1)]  # Corrected line
    b = np.array([100**(i - 1) for i in range(1, n + 1)])
    
    # Define variable bounds
    bounds = [(0, None) for _ in range(n)]
    
    return c, A, b, bounds

def solve_klee_minty_problem(n):
    c, A, b, bounds = klee_minty_problem(n)
    
    # Use linear programming solver to solve the problem
    result = opt.linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    return result

def plot_klee_minty_problem(n):
    _, A, _, _ = klee_minty_problem(n)
    
    # Plot the constraint matrix A
    plt.imshow(A, cmap='gray', interpolation='nearest', aspect='auto')
    plt.title(f'Klee-Minty Problem Constraints (n={n})')
    plt.xlabel('Variables')
    plt.ylabel('Constraints')
    plt.colorbar(label='Constraint Coefficients')
    plt.show()

if __name__ == "__main__":
    n = 100  # You can change the value of n here
    
    # Solve the problem
    result = solve_klee_minty_problem(n)
    
    # Print and explain the results
    print(f"Optimal Solution (x): {result.x}")
    print(f"Optimal Objective Value (Z): {result.fun}")
    
    # Plot the constraint matrix A
    plot_klee_minty_problem(n)
