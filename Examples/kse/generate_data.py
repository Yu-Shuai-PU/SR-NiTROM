import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys
import os
plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
plt.rc('text.latex',preamble=r'\usepackage{amsmath}')
sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

import classes
import fom_class_kse

L = 2 * np.pi
nx = 40
x = np.linspace(0, L, num=nx, endpoint=False)
nu = 4/87

sol_template = np.cos(x)
sol_template_dx = -np.sin(x)

fom = fom_class_kse.KSE(L, nu, nx, sol_template, sol_template_dx)

dx = x[1] - x[0]
dt = 1e-3
T = 10
time = dt * np.linspace(0, int(T/dt), int(T/dt) + 1, endpoint=True)
tstep_kse_fom = fom_class_kse.time_step_kse(fom, time)

nsave = 10
tsave = time[::nsave]

sol_path = "./solutions/"
data_path = "./data/"
os.makedirs(sol_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

#%% # Generate and save trajectory
fname_sol_init = data_path + "sol_init_%03d.npy" # for initial condition of u
fname_sol_init_fitted = data_path + "sol_init_fitted_%03d.npy" # for initial condition of u fitted
fname_sol = sol_path + "sol_%03d.npy" # for u
fname_sol_fitted = sol_path + "sol_fitted_%03d.npy" # for u fitted
fname_weight_sol = sol_path + "weight_sol_%03d.npy"
fname_weight_shift_amount = sol_path + "weight_shift_amount_%03d.npy"
fname_rhs = sol_path + "rhs_%03d.npy" # for du/dt
fname_rhs_fitted = sol_path + "rhs_fitted_%03d.npy" # for du/dt fitted
fname_shift_amount = sol_path + "shift_amount_%03d.npy" # for shifting amount
fname_shift_speed = sol_path + "shift_speed_%03d.npy" # for shifting speed
fname_time = sol_path + "time.npy"

# load the final snapshot of FOM solution and save it as initial condition
# sol_FOM = np.load("./solutions/sol_000.npy")
# np.save(fname_sol_init%0,sol_FOM[:,-1])

#%% # Generate and save initial conditions

# amps = np.array([[-1, 2, 3, -4]])
# n_traj = len(amps)
# uIC = np.zeros((nx, n_traj))
# for k in range (n_traj):
#     uIC[:,k] = amps[k,0] * np.sin(x) + amps[k,1] * np.cos(2 * x) + amps[k,2] * np.cos(3 * x) + amps[k,3] * np.sin(4 * x)

n_sol = 1

pool_inputs = (MPI.COMM_WORLD, n_sol)
pool = classes.mpi_pool(*pool_inputs)

for k in range (pool.my_n_sol):

    sol_idx = k + pool.disps[pool.rank]
    print("Running simulation %d/%d"%(sol_idx,n_sol))
    sol_IC = np.load(fname_sol_init%sol_idx).reshape(-1)
    # sol_IC = -np.sin(x) + 2 * np.cos(2 * x) + 3 * np.cos(3 * x) - 4 * np.sin(4 * x)
    
    sol, tsave = tstep_kse_fom.time_step(sol_IC, nsave)
    sol_fitted, shift_amount = fom.template_fitting(sol, sol_template)
    sol_IC_fitted = sol_fitted[:,0]
    rhs = np.zeros_like(sol)
    rhs_fitted = np.zeros_like(sol_fitted)
    shift_speed = np.zeros_like(shift_amount)
    for j in range (sol.shape[-1]):
        rhs[:,j] = fom.evaluate_fom_rhs(0.0, sol[:,j], np.zeros(sol.shape[0]))
        rhs_fitted[:, j] = fom.shift(rhs[:,j], -shift_amount[j])
        sol_fitted_slice_dx = fom.take_derivative(sol_fitted[:,j], order = 1)
        shift_speed[j] = fom.evaluate_fom_shift_speed(rhs_fitted[:,j], sol_fitted_slice_dx)
    weight_sol = np.mean(np.linalg.norm(sol,axis=0)**2 * dx)
    weight_shift_amount = np.mean((shift_amount - shift_amount[0])**2)

    np.save(fname_sol_init%sol_idx,sol_IC)
    np.save(fname_sol_init_fitted%sol_idx,sol_IC_fitted)
    np.save(fname_sol%sol_idx,sol)
    np.save(fname_sol_fitted%sol_idx,sol_fitted)
    np.save(fname_rhs%sol_idx,rhs)
    np.save(fname_rhs_fitted%sol_idx,rhs_fitted)
    np.save(fname_shift_amount%sol_idx,shift_amount)
    np.save(fname_shift_speed%sol_idx,shift_speed)
    np.save(fname_weight_sol%sol_idx,weight_sol)
    np.save(fname_weight_shift_amount%sol_idx,weight_shift_amount)

np.save(sol_path + "time.npy",tsave)

plt.figure(figsize=(10,6))
plt.contourf(x,tsave,sol.T)
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$t$")
plt.tight_layout()
plt.show()

# print the final shift amount
print("Final shift amount: %.4f"%(shift_amount[-1]))  