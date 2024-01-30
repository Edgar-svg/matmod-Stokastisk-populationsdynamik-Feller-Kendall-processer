#%%
import numpy as np
import scipy as sp
from scipy.linalg import eigh
from matplotlib import pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format='retina'
# %% create specific matrix 1.1
N=51
L = 100
# förra veckans föreläsningar
def createTmatrix(N): # as described in the project pdf
    A = np.zeros((N,N))
    for i in range(0, N):
        A[i-1][i] =  1 # creates superdiagonal of 1s
        A[i][i]   = -2 # creates diagonal of -2s
        A[i][i-1] =  1 # creates subdiagonal of 1s

    # the loop creates 1s at the corners, this removes them
    A[0][N-1]=0    
    A[N-1][0]=0

    return A

def twopBVP(fvec, alpha, beta, L, N):
    dx = L/(N+1)
    fvec[0] = -alpha/dx**2 + fvec[0]
    fvec[N-1] = -beta/dx**2 + fvec[N-1]
    
    T = createTmatrix(N)
    olle = np.array([alpha])
    olle = np.append(olle, np.linalg.solve(1/dx**2*T, fvec))
    olle = np.append(olle, beta)
    return olle

y = lambda x: x**4 / 12
f = lambda x: x**2

def funcEval(f, L, N):
    return np.array(
        [f(x*L/N) for x in range(1, N+1)]
        )

xs = [x*L/N for x in range(0, N+2)]
fvec = funcEval(f, L, N)
ys = twopBVP( fvec, 0, y(L), L, N)

plt.scatter( xs, ys, 
         label='ndiff')
plt.plot(xs, [y(x*L/N) for x in range(0, N+2)], 
         label='teoretical', color='r')
plt.legend()

#%% 1.1 Plotting in error loglog NOT WORKING
error = []
Num_N = 11
N_v = [2**k for k in range(6, Num_N)]
for N in N_v:
    fvec = funcEval(f, L, N)
    dx = L/(N+1)
    ys = twopBVP( fvec, 0, y(L), L, N )
    error.append(ys[N-1]-[y(x*L/N) for x in range(0, N)][N-1])


plt.loglog((error))
plt.scatter(np.log(error), N_v)
plt.grid()

#%% The Beam Equation
N = 999
L = 10
E = 1.9*10**11
I = lambda x: (3-2*np.cos(np.pi*x/L)**12) * 10**(-3)

q = lambda x: -50_000


xs = [x*L/N for x in range(0, N)]
qs = funcEval(q, L, N)
dx = L/(N+1)
ms = twopBVP( qs, 0, 0, L, N )

plt.title('M(x)')
plt.xlabel('meters')
plt.scatter( xs, ms, )

Is = funcEval(I, L, N)
us = twopBVP( ms/E/Is, 0, 0, L, N )
plt.figure()
plt.title('beams centerline deflection u(x)')
plt.scatter(xs, us)
print(f'u(5) is {us[499]} meters')
plt.xlabel('meters')
plt.ylabel('meters')

# %% Part 2. Sturm–Liouville eigenvalue problems
# %% Sturm Lioville solver
# %%

# starting conditions
N = 499
alpha = 0
beta = 0
L = 1
xs = [x*L/N for x in range(0, N)]
qs = funcEval(q, L, N)
dx = L/(N+1)
ms = twopBVP( qs, 0, 0, L, N )


# %% 2.1
N = 499
#theoretical eigenvalues
eig_a = lambda k, L: -((2*k - 1)**2 * np.pi**2/4/L**2)

def eigSolver(alpha, beta, L, N):
    dx = L/N
    T = createTmatrix(N)
    T[-1][-1] = -1
    T = T/dx**2 

    return np.linalg.eig(T)


eig, vec = eigSolver(0, 0, 1, N)

Z = [x for _,x in sorted(zip(eig,vec))]
eig.sort()
Z
plt.plot(Z[-1])
# %%

# %%

# %%

#%%
def error(N, k):
    eig, _ = eigSolver(0, 0, 1, N)
    eig.sort()
    num_eig = eig[-k]
    eig = eig_a(k, L)
    return num_eig - eig
    
print(error(499, 1))
print(error(499, 2))
print(error(499, 3))

#%%
def errVn(k):
    N_vals = [i for i in range(100, 1000, 100)]

    errors = []

    dxs = []

    for N in N_vals:
        err = error(N, k)
        errors.append(err)
        #dx = 1/N
        #dxs.append(dx)

    x = np.logspace(2, 3, 100)
    y = 10**(-2) * x**(-2)
    plt.loglog(x, y, label='y = 10^(-2) * x^(-2)')


    plt.loglog(N_vals, errors)
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.title('Global Error vs. N')
    plt.grid()
    plt.legend()
    plt.show()


errVn(1)

# %% 
eig, vec = eigSolver(0, 0, 1, N)
vec.sort()
print(vec)

# %%

# %% 2.2
N = 499
#theoretical eigenvalues
#eig_a = lambda k, L: -((2*k - 1)**2 * np.pi**2/4/L**2)

def eigSolver(alpha, beta, L, N):
    dx = L/N
    T = createTmatrix(N)
    T[-1][-1] = 0
    T = T/dx**2

    return np.linalg.eig(T)

eig, vec = eigSolver(0, 0, 1, N)

Z = [x for _,x in sorted(zip(eig,vec))]
eig.sort()
Z
plt.plot(Z[-1])
# %%

# %% CHST GGTP
def solve_schrodinger(V, n):
    """
    Solves the stationary Schrödinger equation for a given potential V(x) and returns the first n eigenvalues and eigenfunctions.
    """
    # Define the domain of x
    x = np.linspace(0, 1, 1000)

    # Define the step size
    dx = x[1] - x[0]

    # Define the Laplacian operator
    D2 = np.diag(-2*np.ones(len(x))) + np.diag(np.ones(len(x)-1), 1) + np.diag(np.ones(len(x)-1), -1)
    D2 /= dx**2

    # Define the potential energy operator
    V = np.diag(V(x))

    # Define the Hamiltonian operator
    H = -0.5*D2 + V

    # Compute the eigenvalues and eigenvectors
    E, psi = np.linalg.eigh(H)

    # Normalize the wave functions
    psi /= np.sqrt(dx)

    # Return the first n eigenvalues and eigenfunctions
    return E[:n], psi[:, :n]

def plot_wave_functions(x, psi, E):
    """
    Plots the wave functions and probability densities.
    """
    # Define the normalization factor
    norm = np.sqrt(np.sum(psi**2, axis=0))

    # Normalize the wave functions
    psi /= norm

    # Define the energy levels
    Ek = np.diag(E)

    # Plot the wave functions
    for i in range(psi.shape[1]):
        plt.plot(x, psi[:, i] + E[i], label=f"ψ{i+1}(x) + E{i+1}")

    # Set the plot title and labels
    plt.title("Wave Functions")
    plt.xlabel("x")
    plt.ylabel("ψ(x) + E")
    plt.legend()
    plt.grid()
    plt.show()

def plot_probability_densities(x, psi, E):
    """
    Plots the wave functions and probability densities.
    """
    # Define the normalization factor
    norm = np.sqrt(np.sum(psi**2, axis=0))

    # Normalize the wave functions
    psi /= norm

    # Define the energy levels
    Ek = np.diag(E)

    # Plot the probability densities
    for i in range(psi.shape[1]):
        plt.plot(x, np.abs(psi[:, i])**2 + E[i], label=f"|ψ{i+1}(x)|² + E{i+1}")

    # Set the plot title and labels
    plt.title("Probability Densities")
    plt.xlabel("x")
    plt.ylabel("|ψ(x)|² + E")
    plt.legend()
    plt.grid()
    plt.show()

# Define the potential energy function
def V(x):
    return np.where(x < 0.5, 0, 1)

# Solve the Schrödinger equation
E, psi = solve_schrodinger(V, 6)

# Plot the wave functions and probability densities
x = np.linspace(0, 1, 1000)
plot_wave_functions(x, psi, E)
plot_probability_densities(x, psi, E)


# %%
# Potential function for a particle in a potential box
potential_box = lambda x: 0

# Potential function for a particle encountering a potential barrier
potential_barrier1 = lambda x: 700 * (0.5 - np.abs(x - 0.5))

# Another potential function for a particle encountering a potential barrier
potential_barrier2 = 800 * np.sin(np.pi * x)**2

def solve_schrodinger_equation(potential, num_eigenvalues=6):
    # Discretize the interval [0, 1]
    x = np.linspace(0, 1, 1000)

    T = np.zeros((len(x), len(x)))

    dx = x[1] - x[0]

    for i in range(len(x)):
        T[i, i] = 1 / dx**2 + potential(x[i])
        if i > 0:
            T[i, i - 1] = -1 / (2 * dx**2)
        if i < len(x) - 1:
            T[i, i + 1] = -1 / (2 * dx**2)

    # Solve the eigenvalue problem
    print(T)
    eigenvalues, eigenvectors = eigh(T, eigvals=(0, num_eigenvalues - 1))

    # Normalize the wave functions
    normalized_wavefunctions = eigenvectors / np.sqrt(dx)

    return x, eigenvalues, normalized_wavefunctions

def plot_wavefunctions_and_probabilities(x, eigenvalues, wavefunctions, potential, title):
    num_eigenvalues = len(eigenvalues)
    fig, axs = plt.subplots(num_eigenvalues, 2, figsize=(12, 4 * num_eigenvalues))

    # Plot the potential
    axs[0, 0].plot(x, [potential(xi) for xi in x], label='Potential V(x)')
    axs[0, 0].set_title('Potential V(x)')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('V(x)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot the wave functions and probability densities
    for i in range(num_eigenvalues):
        energy_level = eigenvalues[i]
        wavefunction = wavefunctions[:, i]
        probability_density = np.abs(wavefunction)**2 + energy_level

        # Plot wave function
        axs[i, 1].plot(x, wavefunction + energy_level, label=f'Wavefunction {i + 1}')
        axs[i, 1].set_title(f'Wavefunction {i + 1} + Energy Level')
        axs[i, 1].set_xlabel('x')
        axs[i, 1].set_ylabel('ψ(x) + E')
        axs[i, 1].legend()
        axs[i, 1].grid(True)

        # Plot probability density
        axs[i, 0].plot(x, probability_density, label=f'Probability Density {i + 1}', linestyle='--')
        axs[i, 0].set_title(f'Probability Density {i + 1} + Energy Level')
        axs[i, 0].set_xlabel('x')
        axs[i, 0].set_ylabel('|ψ(x)|^2 + E')
        axs[i, 0].legend()
        axs[i, 0].grid(True)

    fig.suptitle(title)
    plt.show()

# Usage example
x_box, eigenvalues_box, wavefunctions_box = solve_schrodinger_equation(potential_box)
plot_wavefunctions_and_probabilities(x_box, eigenvalues_box, wavefunctions_box, potential_box, 'Particle in a Potential Box')

x_barrier1, eigenvalues_barrier1, wavefunctions_barrier1 = solve_schrodinger_equation(potential_barrier1)
plot_wavefunctions_and_probabilities(x_barrier1, eigenvalues_barrier1, wavefunctions_barrier1, potential_barrier1, 'Potential Barrier 1')

x_barrier2, eigenvalues_barrier2, wavefunctions_barrier2 = solve_schrodinger_equation(potential_barrier2)
plot_wavefunctions_and_probabilities(x_barrier2, eigenvalues_barrier2, wavefunctions_barrier2, potential_barrier2, 'Potential Barrier 2')
# %%

# %%

# %%