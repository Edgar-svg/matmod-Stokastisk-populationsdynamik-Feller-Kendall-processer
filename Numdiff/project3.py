#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

%config InlineBackend.figure_format='retina'
#%%
def eulerstep(Tdx, uold, dt):
    unew = uold + dt*(Tdx @ uold)
    return unew

def creatTdx(L, N):
    dx = L/(N+1)
    sub = np.ones(N-1)
    sup = np.ones(N-1)
    main = -2*np.ones(N)
    T = np.diag(sub, -1) + np.diag(main, 0) + np.diag(sup, 1)
    Tdx = T/(dx**2)
    return Tdx, dx

def solve_diffusion_equation(N, M, g, tend,L):
    Tdx, dx = creatTdx(L, N)
    dt = tend / M
    x = np.linspace(0, L, N)
    xx = np.linspace(0, L, N+2)
    tt = np.linspace(0, tend, M+1)
    T, X = np.meshgrid(tt, xx)

    u = g(x)
    u = u.T

    # Store the solution at each time step
    u_bc = np.pad(u, (1, 1), mode='constant', constant_values=0)
    solution = [u_bc.copy()]

    for n in range(1, M + 1):
        u = eulerstep(Tdx, u, dt)
        u_bc = np.pad(u, (1, 1), mode='constant', constant_values=0)
        solution.append(u_bc.copy())

    
    return np.array(solution), T, X

#%%
L = 1
N = 50
M = 100
tend = 1.0
g = lambda x: np.sin(np.pi * x)

solution, T, X = solve_diffusion_equation(N, M, g, tend, L)
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
T, X = np.meshgrid(np.linspace(0, tend, M + 1), np.linspace(0, L, N + 2))
# Corrected indexing in the plot_surface call


ax.plot_surface(T, X, np.log(solution.T))
ax.set_xlabel('Time')
ax.set_ylabel('Space')
ax.set_zlabel('u(t, x)')
plt.show()
#%%
def crank_nicolson(Tdx, uold, td):
    r, _ = np.shape(Tdx)
    I = np.identity(r)
    A = (I - (td/2)*Tdx)
    B = uold @ (I + (td/2)*Tdx)
    X = np.linalg.solve(A, B)
    return X
#%%
print(X[1,:])
asdf = X[1,:]
print(asdf)