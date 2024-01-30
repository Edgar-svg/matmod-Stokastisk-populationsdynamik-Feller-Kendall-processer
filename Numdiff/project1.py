# %% with(LinearAlgebra)
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format='retina'

# %% All functions
# The RK4step function implements the classic fourth-order Runge-Kutta method for solving ordinary differential equations.
def RK4step(f, told, uold, h):
    # Check if the function returns a scalar
    if np.isscalar(f(told, uold)):
        # Calculate the four stages of the Runge-Kutta method
        Y1 = f(told, uold)
        Y2 = f(told+(h/2),uold+(h/2)*Y1)
        Y3 = f(told+(h/2),uold+(h/2)*Y2)
        Y4 = f(told + h, uold+h*Y3)
        # Return the new value of the solution
        return uold + (h/6)*(Y1 + 2*Y2 + 2*Y3 + Y4)
    else:
        # If the function does not return a scalar, handle it as an array
        Y1 = np.array(f(told, uold))
        Y2 = np.array(f(told+(h/2),uold+(h/2)*Y1))
        Y3 = np.array(f(told+(h/2),uold+(h/2)*Y2))
        Y4 = np.array(f(told + h, uold+h*Y3))
        # Return the new value of the solution
        return np.array(uold) + (h/6) * (Y1 + 2*Y2 + 2*Y3 + Y4)

# The RK34step function implements a variant of the Runge-Kutta method that includes an error estimate.
def RK34step(f, told, uold, h):
    # Check if the function returns a scalar
    if np.isscalar(f(told, uold)):
        # Calculate the four stages of the Runge-Kutta method
        Y1 = f(told, uold)
        Y2 = f(told+(h/2),uold+(h/2)*Y1)
        Y3 = f(told+(h/2),uold+(h/2)*Y2)
        Y4 = f(told + h, uold+h*Y3)
        # Calculate an additional stage for error estimation
        Z3 = f(told + h, uold - h*Y1 + 2*h*Y2)
        # Calculate the new value of the solution using RK4step
        unew = RK4step(f, told, uold, h)
        # Calculate the error estimate
        err = (h/6)*(2*Y2 + Z3 - 2*Y3 - Y4)
    else:
        # If the function does not return a scalar, handle it as an array
        Y1 = np.array(f(told, uold))
        Y2 = np.array(f(told+(h/2),uold+(h/2)*Y1))
        Y3 = np.array(f(told+(h/2),uold+(h/2)*Y2))
        Y4 = np.array(f(told + h, uold+h*Y3))
        # Calculate an additional stage for error estimation
        Z3 = np.array(f(told + h, uold - h*Y1 + 2*h*Y2))
        # Calculate the new value of the solution using RK4step
        unew = np.array(RK4step(f, told, uold, h))
        # Calculate the error estimate
        err = (h/6)*np.array((2*Y2 + Z3 - 2*Y3 - Y4))
    # Return the new value of the solution and the error estimate
    return unew, err

# The newstep function calculates the new step size based on the current error, the previous error, the old step size, and a constant k.
def newstep(tol, err, errold, hold, k):
    # Calculate the norms of the current and previous errors
    errn = np.linalg.norm(err)
    errnold = np.linalg.norm(errold)
    # Return the new step size
    return (tol/errn)**(2/(3*k))*(tol/errnold)**(-1/(3*k))*hold

# The adaptiveRK34 function implements an adaptive Runge-Kutta method for solving ordinary differential equations.
def adaptiveRK34(f, t0, tf, y0, tol):
    # Calculate the initial step size
    h0 = ((np.abs(tf-t0))*(tol**(1/4)))/(100*(1+(np.linalg.norm(f(t0, y0)))))
    h = h0
    t = t0
    t_vec = [t]
    y = [y0]

    # Perform the first step
    ynew, err = RK34step(f, t, y[-1], h)
    hold = h
    t += hold
    t_vec.append(t)
    y.append(ynew)
    # Calculate the new step size
    h = newstep(tol, err, tol, hold, 4)
    errold = err
    
    # Perform the remaining steps
    while t < tf:
        ynew, err = RK34step(f, t, y[-1], h)
        hold = h
        h = newstep(tol, err, errold, hold, 4)
        errold = err
        y.append(ynew)
        t += h
        t_vec.append(t)

    # Adjust the final step to reach exactly tf
    t -= h
    h = tf - t
    ynew, err = RK34step(f, t, y[-1], h)
    t_vec[-1] = t + h
    y[-1] = ynew
    
    # Return the solution
    if np.isscalar(f(t0, y0)):
        return t_vec, y
    else:
        return np.array(t_vec), np.array(y)

# The RK4int function implements the fourth-order Runge-Kutta method with a fixed step size.
def RK4int(f, y0, t0, tf, N):
    # Generate the grid of time points
    tgrid = np.linspace(t0, tf, N+1)
    h = (tf-t0)/N
    approx = [y0]
    err = []
    lam = f(0,1)
    # Calculate the exact solution for comparison
    if np.isscalar(lam):
        exact_solution = y0 * np.exp(lam * (tf - t0))
    else:
        exact_solution = sp.linalg.expm(lam *(tf - t0)) @ y0
    # Perform the steps
    for told in tgrid:
        yold = approx[-1]
        ynew = RK4step(f, told, yold, h)
        approx.append(ynew)
        # Calculate the error at each step
        err.append(np.linalg.norm(approx[-1] - exact_solution))
    # Return the grid, the approximate solution, and the errors
    return tgrid, approx, err

# The errVSh function plots the global error versus the step size for the RK4int method.
def errVSh(f, y0, t0, tf):
    h_values = []

    num_N = 20
    N_values = [2**k for k in range(9,num_N)]

    errors = []

    # Calculate the errors for different step sizes
    for N in N_values:
        tgrid, approx, err = RK4int(f, y0, t0, tf, N)
        errors.append(err[-1])
        h_values.append((tf - t0) / N)
    print(errors)

    # Plot the errors
    plt.loglog(h_values, errors)

    plt.xlabel('Step Size (h)')
    plt.ylabel('Error')
    plt.title('Global Error vs. Step Size')
    plt.grid()
    plt.show()

# Define the function to be integrated
function = lambda t, y: -1*y

errVSh(function, 1, 0, 100)
#%% 1
tol = 1e-3
tol2= 1e-12
t0 = 0 
tf = 10
y0 = 1
N = 20

t, y = adaptiveRK34(function, t0, tf, y0, tol)

t2, y2 = adaptiveRK34(function, t0, tf, y0, tol2)

exact_solution = y0 * np.exp(-(1*tf-t0))

t = np.array(t)

print(f'Globall error at the last point = {np.linalg.norm(y[-1]-exact_solution)}.')
plt.scatter(t, y, label='Approx, tol = 1e-3')
plt.scatter(t2, y2, label='Approx, tol = 1e-12')
plt.plot(t, np.exp(-(1*t-t0)), label='Exact')
#plt.yscale("log")
#plt.xscale("log")
plt.xlabel('Time')
plt.ylabel('$y_n$')
plt.legend()
plt.title("$y_n$ over time with adaptive step size")
plt.grid()
plt.show()


#plt.plot(t, errs)
#plt.xlabel('Time')
#plt.ylabel('Error')
#plt.title("Error over time with adaptive step size")
#plt.grid()
#plt.show()



    

# %% 2.1 / 2a 
y0 = np.array([1, 1])
tol = 1e-12
t0 = 0
tf = 10

def lotka(t, u):
    a = 3
    b = 9
    c = 15
    d = 15

    # a = 1
    # b = 2
    # c = 3
    # d = 4

    dxdt = a * u[0] - b * u[0] * u[1]
    dydt = c * u[0] * u[1] - d * u[1]

    return np.array([dxdt, dydt])


t, y = adaptiveRK34(lotka, t0, tf, y0, tol)

# Population dynamics over time
plt.plot(t, y[:, 0], label='Rabbits (x)')
plt.plot(t, y[:, 1], label='Foxes (y)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Population dynamics over time')
plt.grid()
plt.show()

# Phase Portrait
plt.plot(y[:, 0], y[:, 1])
plt.xlabel('Rabbits (x)')
plt.ylabel('Foxes (y)')
plt.title('Phase Portrait')
plt.grid()
plt.show()



#%% 2c

def hamiltonian(x, y):
    a = 3
    b = 9
    c = 15
    d = 15
    return c * x + b * y - d * np.log(x) - a * np.log(y)

tf = 500

t, y = adaptiveRK34(lotka, t0, tf, y0, tol)

# Calculate the Hamiltonian values
H0 = hamiltonian(y0[0], y0[1])
H_values = [hamiltonian(xi, yi) for xi, yi in zip(y[:, 0], y[:, 1])]

# Calculate the relative error in Hamiltonian
relative_error = np.abs(np.array(H_values) / H0 - 1)

# Plot the relative error as a function of time
plt.plot(t, relative_error)
plt.xlabel('Time')
plt.ylabel('|H(x, y)/H(x0, y0) - 1|')
plt.yscale('log')
plt.title('Relative Error in Hamiltonian over Time')
plt.grid()
plt.show()





# %% 3
# Define the van der Pol equation
def van_der_pol(t, u):
    return [u[1], 100 * (1 - u[0]**2) * u[1] - u[0]]

# Implement your adaptive RK34 solver (similar to the one you used for Lotka-Volterra)

# Set initial conditions and parameters
t0 = 0
tf = 200
y0 = [2, 0]  # Initial condition
mu = 100

# Call your adaptive RK34 solver
t, y = adaptiveRK34(van_der_pol, t0, tf, y0, tol=1e-6)


# Plot y2 as a function of time
plt.plot(t, y[:, 1], label='y2')
plt.xlabel('Time')
plt.ylabel('y2')
plt.legend()
plt.title('van der Pol Equation for μ = 100')
plt.grid()
plt.show()

# Plot y2 as a function of y1 (Phase Portrait)
plt.plot(y[:, 0], y[:, 1])
plt.xlabel('y1')
plt.ylabel('y2')
plt.title('Phase Portrait for van der Pol Equation (μ = 100)')
plt.grid()
plt.show()


#%% 3.2
def adaptiveRK34_mu(t0, tf, y0, tol, mu):
    #mu_values = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000]

    num_steps = []
 
    def van_der_pol_mu(t, u):
        return [u[1], mu * (1 - u[0]**2) * u[1] - u[0]]
    t, y = adaptiveRK34(van_der_pol_mu, t0, tf, y0, tol)
    num_steps.append(len(t))
    return t, y, num_steps
            

# Solve for the E6 series of values of μ
mu_values = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470]
num_steps_values = []

for mu in mu_values:
    _, _, num_steps = adaptiveRK34_mu(t0, 0.7 * mu, y0, tol=1e-6, mu=mu)
    num_steps_values.append(num_steps)

# Plot the total number of steps as a function of μ in a loglog diagram
plt.loglog(mu_values, num_steps_values, marker='o', linestyle='-', color='b')
plt.xlabel('μ')
plt.ylabel('Number of Steps')
plt.title('Number of Steps vs. Stiffness for van der Pol Equation')
plt.grid()
plt.show()


#%% 3.3 Mine vs professional code
from scipy.integrate import solve_ivp

# Solve for the E6 series of values of μ using solve_ivp with BDF method
num_steps_ivp_values = []

for mu in mu_values:
    sol = solve_ivp(van_der_pol, [t0, 0.7 * mu], y0, method='BDF', atol=1e-6, rtol=1e-6, args=(mu,))
    num_steps_ivp_values.append(len(sol.t))

# Plot the total number of steps as a function of μ in a loglog diagram
plt.loglog(mu_values, num_steps_values, marker='o', linestyle='-', color='b', label='Adaptive RK34')
plt.loglog(mu_values, num_steps_ivp_values, marker='s', linestyle='--', color='r', label='solve_ivp (BDF)')
plt.xlabel('μ')
plt.ylabel('Number of Steps')
plt.title('Number of Steps vs. Stiffness for van der Pol Equation')
plt.legend()
plt.grid()
plt.show()


# %%
