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

max_wavenumber_stable = int(np.ceil(np.sqrt(1/nu)))
# sol_template = np.zeros_like(x)
# sol_template_dx = np.zeros_like(x)
# sol_template_dxx = np.zeros_like(x)
# for wavenumber_idx in range(1, max_wavenumber_stable + 1):
#     sol_template += np.cos(wavenumber_idx * x) / wavenumber_idx + np.sin(wavenumber_idx * x) / wavenumber_idx
#     sol_template_dx += -np.sin(wavenumber_idx * x) + np.cos(wavenumber_idx * x)
#     sol_template_dxx += -np.cos(wavenumber_idx * x) * wavenumber_idx - np.sin(wavenumber_idx * x) * wavenumber_idx
    
X_template = np.cos(x)
X_template_dx = -np.sin(x)
X_template_dxx = -np.cos(x)

fom = fom_class_kse.KSE(L, nu, nx, X_template, X_template_dx)
dx = x[1] - x[0]
dt = 1e-3
T = 4
time = dt * np.linspace(0, int(T/dt), int(T/dt) + 1, endpoint=True)
tstep_kse_fom = fom_class_kse.time_step_kse(fom, time)

nsave = 10
tsave = time[::nsave]
start_time = 80

traj_path = "./trajectories/"
data_path = "./data/"
fig_path = "./figures/"
os.makedirs(traj_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)

#%% # Generate and save trajectory
fname_X_template = data_path + "X_template.npy"
fname_X_template_dx = data_path + "X_template_dx.npy"
fname_X_template_dxx = data_path + "X_template_dxx.npy"
fname_traj_init = data_path + "traj_init_%03d.npy" # for initial condition of u
fname_traj_init_fitted = data_path + "traj_init_fitted_%03d.npy" # for initial condition of u fitted
fname_traj = traj_path + "traj_%03d.npy" # for u
fname_traj_fitted = traj_path + "traj_fitted_%03d.npy" # for u fitted
fname_weight_traj = traj_path + "weight_traj_%03d.npy"
fname_weight_shift_amount = traj_path + "weight_shift_amount_%03d.npy"
fname_weight_shift_speed = traj_path + "weight_shift_speed_%03d.npy"
fname_rhs = traj_path + "deriv_%03d.npy" # for du/dt
fname_rhs_fitted = traj_path + "deriv_fitted_%03d.npy" # for du/dt fitted
fname_shift_amount = traj_path + "shift_amount_%03d.npy" # for shifting amount
fname_shift_speed = traj_path + "shift_speed_%03d.npy" # for shifting speed
fname_time = traj_path + "time.npy"

# The initial condition is generated as follows:
# 1. start from an initial condition
#     u = -sin(x) + 2cos(2x) + 3cos(3x) - 4sin(4x)
# 2. stop at specific time (post transient, here we choose t_start = 80)
# 3. record this snapshot as the base initial condition

n_traj = 9
traj_IC_array = np.zeros((nx, n_traj))
traj_IC_original = np.loadtxt(f"initial_condition_time_{start_time}.txt").reshape(-1) 
amp = 1.0
amp_array = np.array([-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0]) * amp
cmap_name = 'bwr'

traj_IC_array[:, 0] = traj_IC_original
for idx_k, k in enumerate(amp_array):
    traj_IC_array[:, idx_k + 1] = traj_IC_original + k * np.sin(2 * np.pi / L * x)
    print(f"initial condition No.{idx_k + 1}, perturbation amp proportion = {np.sqrt(np.linalg.norm(traj_IC_array[:, idx_k + 1] - traj_IC_original)**2 / np.linalg.norm(traj_IC_original)**2):.4f}")

# for k in range (1, n_traj):
#     # Here we want to use random perturbations
#     np.random.seed(k)
#     random_perturbation = np.random.randn(nx)
#     random_perturbation = random_perturbation - np.mean(random_perturbation) # make sure the perturbation has zero mean
#     traj_IC_array[:, k] = traj_IC_original + np.sqrt(nx) * random_perturbation * amp / np.linalg.norm(random_perturbation)

# load the final snapshot of FOM solution and save it as initial condition
# sol_FOM = np.load("./solutions/sol_000.npy")
# np.save(fname_sol_init%0,sol_FOM[:,-1])
# sol_init = sol_FOM[:,-1]
# np.savetxt(data_path + "initial_condition_time_80.txt",sol_init)

#%% # Generate and save initial conditions

np.save(fname_X_template, X_template)
np.save(fname_X_template_dx, X_template_dx)
np.save(fname_X_template_dxx, X_template_dxx)
np.save(traj_path + "time.npy",tsave)

pool_inputs = (MPI.COMM_WORLD, n_traj)
pool = classes.mpi_pool(*pool_inputs)
for k in range (pool.my_n_traj):
    traj_idx = k + pool.disps[pool.rank]
    print("Running simulation %d/%d"%(traj_idx,n_traj - 1))
    traj_IC = traj_IC_array[:,traj_idx]

    traj, tsave = tstep_kse_fom.time_step(traj_IC, nsave)
    traj_fitted, shift_amount = fom.template_fitting(traj, X_template)
    traj_IC_fitted = traj_fitted[:,0]
    deriv = np.zeros_like(traj)
    deriv_fitted = np.zeros_like(deriv)
    shift_speed = np.zeros_like(shift_amount)
    shift_speed_numer = np.zeros_like(shift_amount)
    shift_speed_denom = np.zeros_like(shift_amount)
    for j in range (traj.shape[-1]):
        deriv[:,j] = fom.evaluate_fom_rhs(0.0, traj[:,j], np.zeros(traj.shape[0]))
        deriv_fitted[:, j] = fom.shift(deriv[:,j], -shift_amount[j])
        traj_fitted_slice_dx = fom.spatial_deriv(traj_fitted[:,j], order = 1)
        shift_speed[j] = fom.evaluate_fom_shift_speed(deriv_fitted[:,j], traj_fitted_slice_dx)
        shift_speed_numer[j] = fom.evaluate_fom_shift_speed_numer(deriv_fitted[:,j])
        shift_speed_denom[j] = fom.evaluate_fom_shift_speed_denom(traj_fitted_slice_dx)
    weight_traj = 1.0/np.mean(np.linalg.norm(traj,axis=0)**2)
    weight_shift_amount = 1.0/np.mean((shift_amount - shift_amount[0])**2)
    weight_shift_speed = 1.0/np.mean(shift_speed**2)

    np.save(fname_traj_init%traj_idx,traj_IC)
    np.save(fname_traj_init_fitted%traj_idx,traj_IC_fitted)
    np.save(fname_traj%traj_idx,traj)
    np.save(fname_traj_fitted%traj_idx,traj_fitted)
    np.save(fname_rhs%traj_idx,deriv)
    np.save(fname_rhs_fitted%traj_idx,deriv_fitted)
    np.save(fname_shift_amount%traj_idx,shift_amount)
    np.save(fname_shift_speed%traj_idx,shift_speed)
    np.save(fname_weight_traj%traj_idx,weight_traj)
    np.save(fname_weight_shift_amount%traj_idx,weight_shift_amount)
    np.save(fname_weight_shift_speed%traj_idx,weight_shift_speed)
    
    ### Plotting, things to be done:
    ### 1. switch from contourf to pcolormesh
    ### 2. apply Fourier spectral interpolation to plot a 40-mode solution on a 256-point grid for better visualization
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,tsave,traj.T, cmap=cmap_name)
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    if traj_idx == 0:
        plt.title(f"FOM solution, initial condition = uIC")
    else:   
        plt.title(f"FOM solution, initial condition = uIC + {amp} * {amp_array[traj_idx - 1]} * sin(x)")
    plt.savefig(fig_path + "sol_FOM_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,tsave,traj_fitted.T, cmap=cmap_name)
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    if traj_idx == 0:
        plt.title(f"FOM solution (fitted), initial condition = uIC")
    else:
        plt.title(f"FOM solution (fitted), initial condition = uIC + {amp} * {amp_array[traj_idx - 1]} * sin(x)")

    plt.savefig(fig_path + "sol_FOM_fitted_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(tsave,shift_amount)
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift amount $c(t)$")
    plt.tight_layout()

    if traj_idx == 0:
        plt.title(f"Shift amount, initial condition = uIC")
    else:
        plt.title(f"Shift amount, initial condition = uIC + {amp} * {amp_array[traj_idx - 1]} * sin(x)")
    
    plt.savefig(fig_path + "shift_amount_FOM_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(tsave,shift_speed_numer, label="numerator")
    plt.plot(tsave,shift_speed_denom, label="denominator")
    plt.plot(tsave,shift_speed, label="shift speed")
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift speed $c'(t)$")
    plt.legend()
    plt.tight_layout()

    if traj_idx == 0:
        plt.title(f"Shift speed, initial condition = uIC")
    else:
        plt.title(f"Shift speed, initial condition = uIC + {amp} * {amp_array[traj_idx - 1]} * sin(x)")
    
    plt.savefig(fig_path + "shift_speed_FOM_%03d.png"%traj_idx)
    plt.close()
    
    ## I want to plot the Fourier components of the solution with time
    ## Plot the Fourier components of the solution with time
    # Perform Fourier transform along the spatial axis for all time steps
    traj_fft_time = np.fft.fft(traj, axis=0)
    
    # Calculate the corresponding integer wavenumbers k
    # For a domain of length L, the wavenumbers are k = n * (2*pi/L). Here L=2*pi, so k=n.
    wavenumbers = np.fft.fftfreq(nx, d=dx) * L

    # This reorders the arrays from the standard FFT output to a centered order.
    wavenumbers_shifted = np.fft.fftshift(wavenumbers)
    traj_fft_time_shifted = np.fft.fftshift(traj_fft_time, axes=0)
    plt.figure(figsize=(10,6))
    plt.pcolormesh(wavenumbers_shifted, tsave, np.abs(traj_fft_time_shifted).T, shading='auto', vmin=0)
    # plt.contourf(wavenumbers_shifted, tsave, np.abs(traj_fft_time_shifted.T))
    plt.colorbar()
    plt.xlabel(r"wavenumber $\xi$")
    plt.ylabel(r"time $t$")

    if traj_idx == 0:
        plt.title(f"Fourier components of the solution, initial condition = uIC")
    elif 1 <= traj_idx <= (n_traj - 1) // 2:
        plt.title(f"Fourier components of the solution, initial condition = uIC + {amp} * cos({traj_idx} * x)")
    else:
        plt.title(f"Fourier components of the solution, initial condition = uIC + {amp} * sin({traj_idx - (n_traj-1)//2} * x)")
    
    plt.savefig(fig_path + "Fourier_components_FOM_%03d.png"%traj_idx)
    plt.close()

    # print the final shift amount
    print("Final shift amount: %.4f"%(shift_amount[-1]))
    
    # print the L2 norm of the initial condition
    print(f"rank = {pool.rank}, L2 norm of the initial condition: {np.linalg.norm(traj_IC_array[:, traj_idx])}")


