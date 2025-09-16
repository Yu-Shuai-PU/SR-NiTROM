import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys
import os

sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

import classes
import fom_class_kse

L = 2 * np.pi
nx = 40
x = np.linspace(0, L, num=nx, endpoint=False)
nu = 4/87

fom = fom_class_kse.KSE(L, nu, nx)

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
fname_sol = sol_path + "sol_%03d.npy" # for u
fname_sol_fitted = sol_path + "sol_fitted_%03d.npy" # for u fitted
fname_weight = sol_path + "weight_%03d.npy"
fname_rhs = sol_path + "rhs_%03d.npy" # for du/dt
fname_rhs_fitted = sol_path + "rhs_fitted_%03d.npy" # for du/dt fitted
fname_shift_amount = sol_path + "shift_amount_%03d.npy" # for shifting amount
fname_shift_speed = sol_path + "shift_speed_%03d.npy" # for shifting speed
fname_time = sol_path + "time.npy"

# amps = np.array([[-1, 2, 3, -4]])
# n_traj = len(amps)
# uIC = np.zeros((nx, n_traj))
# for k in range (n_traj):
#     uIC[:,k] = amps[k,0] * np.sin(x) + amps[k,1] * np.cos(2 * x) + amps[k,2] * np.cos(3 * x) + amps[k,3] * np.sin(4 * x)

n_sol = 1

pool_inputs = (MPI.COMM_WORLD, n_sol, fname_time)
pool_kwargs = {'fname_sol':fname_sol,'fname_sol_fitted':fname_sol_fitted,
               'fname_rhs':fname_rhs,'fname_rhs_fitted':fname_rhs_fitted,
               'fname_shift_amount':fname_shift_amount,'fname_shift_speed':fname_shift_speed,
               'fname_weights':fname_weight}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)

u_IC = np.zeros((nx, pool.my_n_sol))

u_template = np.cos(x)
u_template_dx = -np.sin(x)

for k in range (pool.my_n_sol):

    sol_idx = k + pool.disps[pool.rank]
    print("Running simulation %d/%d"%(sol_idx,n_sol))
    u_IC[:, k] = np.load(fname_sol_init%sol_idx).reshape(-1)

    sol, tsave = tstep_kse_fom.time_step(u_IC[:,k],nsave)
    sol_fitted, shift_amount = fom_class_kse.template_fitting(sol, u_template, L, nx)
    rhs = np.zeros_like(sol)
    rhs_fitted = np.zeros_like(sol_fitted)
    shift_speed = np.zeros_like(shift_amount)
    for j in range (sol.shape[-1]):
        rhs[:,j] = fom.evaluate_fom_rhs(0.0, sol[:,j], np.zeros(sol.shape[0]))
        rhs_fitted[:, j] = fom_class_kse.shift(rhs[:,j], -shift_amount[j], L)
        sol_fitted_slice_dx = fom.take_derivative(sol_fitted[:,j], order = 1)
        shift_speed[j] = fom_class_kse.compute_shift_speed_FOM(rhs_fitted[:,j], sol_fitted_slice_dx, u_template_dx)
    weight = np.mean(np.linalg.norm(rhs,axis=0)**2)

    np.save(fname_sol%sol_idx,sol)
    np.save(fname_sol_fitted%sol_idx,sol_fitted)
    np.save(fname_rhs%sol_idx,rhs)
    np.save(fname_rhs_fitted%sol_idx,rhs_fitted)
    np.save(fname_shift_amount%sol_idx,shift_amount)
    np.save(fname_shift_speed%sol_idx,shift_speed)
    np.save(fname_weight%sol_idx,weight)

np.save(sol_path + "time.npy",tsave)