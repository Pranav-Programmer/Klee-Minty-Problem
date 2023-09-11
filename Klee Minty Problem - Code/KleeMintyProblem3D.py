import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def plot_klee_minty_problem_3d(n):
    _, A, _, _ = klee_minty_problem(n)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(n):
        x = np.zeros(n)
        x[:i+1] = 1  # Set some variables to 1 to visualize constraints
        z = np.dot(A, x)
        y = np.arange(n) + 1  # Constraint indices
        ax.bar(y, z, zs=i, zdir='y', width=0.5, alpha=0.8)
    
    ax.set_xlabel('Constraints')
    ax.set_ylabel('Variables')
    ax.set_zlabel('Constraint Coefficients')
    ax.set_title(f'Klee-Minty Problem Visualization (n={n})')
    
    plt.show()

if __name__ == "__main__":
    n = 3  # You can change the value of n here
    
    # Plot the 3D visualization
    plot_klee_minty_problem_3d(n)
