import numpy as np
import scipy.linalg as sp
from matplotlib import pyplot as plt

def eulerstep(A, uold, h):
    if np.isscalar(A):
        return uold + h * A * uold
    else:
        return uold + h * (A @ uold)

def eulerint(A, y0, t0, tf, N):
    y0 = np.asarray(y0)
    tgrid = np.linspace(t0, tf, N+1)
    h = (tf-t0)/N
    approx = np.array([y0])
    err = []
    if np.isscalar(A):
        exact_solution = y0 * np.exp(A * (tf - t0))
        err.append(np.linalg.norm(approx[-1] - exact_solution))
        for i in tgrid[1:]:
            yold = approx[-1]
            ynew = eulerstep(A, yold, h)
            approx = np.append(approx, [ynew], axis=0)
            err.append(np.linalg.norm(approx[-1] - exact_solution))
        
    else:
        A = np.asarray(A)
        exact_solution = (sp.expm(A * (tf - t0))) @ y0
        err.append(np.linalg.norm(approx[-1] - exact_solution))
        for _ in tgrid[1:]:
            yold = approx[-1]
            ynew = eulerstep(A, yold, h)
            approx = np.append(approx, [ynew], axis=0)
            err.append(np.linalg.norm(approx[-1] - exact_solution))    
    return tgrid, approx, err

def errVSh(A, y0, t0, tf):
    h_values = []

    num_N = 15
    N_values = [2**k for k in range(1, num_N)]

    errors = []

    for N in N_values:
        tgrid, approx, err = eulerint(A, y0, t0, tf, N)
        errors.append(err[-1])
        h_values.append((tf - t0) / N)

    plt.loglog(h_values, errors)

    plt.xlabel('Step Size (h)')
    plt.ylabel('Error')
    plt.title('Global Error vs. Step Size')
    plt.grid()
    plt.show()



A_scalar = -1
y0_scalar = 1
t0_scalar = 0
tf_scalar = 1

N = 10000

A_matrix = [[-1, 10], [0, -3]]
y0_matrix = [[1], [1]]
t0_matrix = 0
tf_matrix = 10

#tgride, approxx, err =eulerint(A_scalar, y0_scalar, t0_scalar, tf_scalar, N)
tgride, approxx, err = eulerint(A_matrix, y0_matrix, t0_matrix, tf_matrix, N)

plt.plot(tgride, err)
plt.yscale("log")
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Global Error vs. Time')
plt.grid()
plt.show()

#errVSh(A_scalar, y0_scalar, t0_scalar, tf_scalar)
#errVSh(A_matrix, y0_matrix, t0_matrix, tf_matrix)
