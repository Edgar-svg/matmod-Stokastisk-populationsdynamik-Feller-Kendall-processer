import scipy as sp 
from matplotlib import pyplot as plt
import numpy as np 
k=-1
f = lambda x, t: k*x + 0*t

def RK4step(f, told, yold, h):
    k1 = f(told,     yold)
    k2 = f(told+h/2, yold+h*k1/2)
    k3 = f(told+h/2, yold+h*k2/2)
    k4 = f(told+h,   yold+h*k3)
    
    # Combine partial steps.
    ynew = yold + h/6*(k1+2*k2+2*k3+k4)  
    return ynew 

def RK3step(f, told, yold, h):
    k1 = f(told,     yold)
    k2 = f(told+h/2, yold+h*k1/2)
    k3 = f(told+h,   yold-h*k1 +2*h*k2)
    
    # Combine partial steps.
    ynew = yold + h/6*(k1+4*k2+k3)  
    return ynew 

def local_error(f, told, yold, h):
    k1 = f(told,     yold)
    k2 = f(told+h/2, yold+h*k1/2)
    k3 = f(told+h/2, yold+h*k2/2)
    z3 = f(told+h,   yold-h*k1 +2*h*k2)
    k4 = f(told+h,   yold+h*k3)

    error = h/6 *(2*k2+z3-2*k3 -k4)
    return error

def RK34step(f, told, uold, h):
    unew = RK4step(f, told, yold, h)
    error = local_error(f, told, yold, h)
    return unew, error 

"""
given the tolerance tol, 
a local error estimate err and a previous error estimate errold, 
the old step size hold, and the order k of the error estimator, 
computes the new step size hnew using (1).
"""
def newstep(tol,err, errold, hold, k):
    a = tol/err 
    b = tol/errold 
    hnew = a**(2/3/k) * b**(-1/3/k) * hold
    return hnew 

"""
solves y′ = f(t,y); y(t0) = y0 on the interval [t0,tf], 
while keeping the error estimate equal to tol
using the step size control algorithm implemented in newstep.
In the vector t you store the time points the method uses, 
and in y you store the corresponding numerical approximation,
as a row vector for each value of t.
"""
def adaptiveRK34(f, t0, tf, y0, tol):
    error0 = tol 
    unew, error = RK34step(f, told, uold, h)
    hnew = newstep(tol, error, errold, hold, k)




def RK4int(f, y0, t0, tf, N):
    tgrid = np.linspace(t0, tf, N+1)
    h = (tf-t0)/N
    approx = np.array([y0])
    err = []
    told = t0
    exact_solution = y0 * np.exp(k*(tf - t0))
    err.append(np.linalg.norm(approx[-1] - exact_solution))

    for i in tgrid[1:]:
        yold = approx[-1]
        ynew = RK4step(f, told, yold, h)
        told += h 

        approx = np.append(approx, [ynew], axis=0)
        exact_solution = y0 * np.exp(k*(i - t0))  # calculate exact solution for each time step
        err.append(np.linalg.norm(approx[-1] - exact_solution))
        

    return tgrid, approx, err

def errVSh(f, y0, t0, tf):
    h_values = []

    num_N = 15
    N_values = [2**k for k in range(6, num_N)]

    errors = []

    for N in N_values:
        tgrid, approx, err = RK4int(f, y0, t0, tf, N)
        errors.append(err[-1])
        h_values.append((tf - t0) / N)

    plt.loglog(h_values, errors)

    plt.xlabel('Step Size (h)')
    plt.ylabel('Error')
    plt.title('Global Error vs. Step Size')
    plt.grid()
    plt.show()

errVSh(f, 1, 0, 100)

