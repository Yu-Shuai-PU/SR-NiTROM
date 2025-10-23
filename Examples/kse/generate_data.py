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
sol_template = np.zeros_like(x)
sol_template_dx = np.zeros_like(x)
sol_template_dxx = np.zeros_like(x)
for wavenumber_idx in range(1, max_wavenumber_stable + 1):
    sol_template += np.cos(wavenumber_idx * x) / wavenumber_idx + np.sin(wavenumber_idx * x) / wavenumber_idx
    sol_template_dx += -np.sin(wavenumber_idx * x) + np.cos(wavenumber_idx * x)
    sol_template_dxx += -np.cos(wavenumber_idx * x) * wavenumber_idx - np.sin(wavenumber_idx * x) * wavenumber_idx
    
sol_template = np.cos(x)
sol_template_dx = -np.sin(x)
sol_template_dxx = -np.cos(x)

fom = fom_class_kse.KSE(L, nu, nx, sol_template, sol_template_dx)

dx = x[1] - x[0]
dt = 1e-3
T = 10
time = dt * np.linspace(0, int(T/dt), int(T/dt) + 1, endpoint=True)
tstep_kse_fom = fom_class_kse.time_step_kse(fom, time)

nsave = 10
tsave = time[::nsave]
start_time = 80

sol_path = "./solutions/"
data_path = "./data/"
fig_path = "./figures/"
os.makedirs(sol_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)

#%% # Generate and save trajectory
fname_sol_template = data_path + "sol_template.npy"
fname_sol_template_dx = data_path + "sol_template_dx.npy"
fname_sol_template_dxx = data_path + "sol_template_dxx.npy"
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

n_sol = 9
sol_IC_array = np.zeros((nx, n_sol))
sol_IC_original = np.loadtxt(data_path + f"initial_condition_time_{start_time}.txt").reshape(-1)
amp = 1.0

sol_IC_array[:, 0] = sol_IC_original

for k in range (1, 1 + (n_sol - 1) // 2):

    sol_IC_array[:, k] = sol_IC_original + amp * np.cos(k * x)
    sol_IC_array[:, k + (n_sol - 1) // 2] = sol_IC_original + amp * np.sin(k * x)

# for k in range (1, n_sol):
#     # Here we want to use random perturbations
#     np.random.seed(k)
#     random_perturbation = np.random.randn(nx)
#     random_perturbation = random_perturbation - np.mean(random_perturbation) # make sure the perturbation has zero mean
#     sol_IC_array[:, k] = sol_IC_original + np.sqrt(nx) * random_perturbation * amp / np.linalg.norm(random_perturbation)

# load the final snapshot of FOM solution and save it as initial condition
# sol_FOM = np.load("./solutions/sol_000.npy")
# np.save(fname_sol_init%0,sol_FOM[:,-1])
# sol_init = sol_FOM[:,-1]
# np.savetxt(data_path + "initial_condition_time_80.txt",sol_init)

#%% # Generate and save initial conditions

np.save(fname_sol_template, sol_template)
np.save(fname_sol_template_dx, sol_template_dx)
np.save(fname_sol_template_dxx, sol_template_dxx)

pool_inputs = (MPI.COMM_WORLD, n_sol)
pool = classes.mpi_pool(*pool_inputs)
for k in range (pool.my_n_sol):
    sol_idx = k + pool.disps[pool.rank]
    print("Running simulation %d/%d"%(sol_idx,n_sol - 1))
    sol_IC = sol_IC_array[:,sol_idx]

    sol, tsave = tstep_kse_fom.time_step(sol_IC, nsave)
    sol_fitted, shift_amount = fom.template_fitting(sol, sol_template)
    sol_IC_fitted = sol_fitted[:,0]
    rhs = np.zeros_like(sol)
    rhs_fitted = np.zeros_like(sol_fitted)
    shift_speed = np.zeros_like(shift_amount)
    shift_speed_numer = np.zeros_like(shift_amount)
    shift_speed_denom = np.zeros_like(shift_amount)
    for j in range (sol.shape[-1]):
        rhs[:,j] = fom.evaluate_fom_rhs(0.0, sol[:,j], np.zeros(sol.shape[0]))
        rhs_fitted[:, j] = fom.shift(rhs[:,j], -shift_amount[j])
        sol_fitted_slice_dx = fom.take_derivative(sol_fitted[:,j], order = 1)
        shift_speed[j] = fom.evaluate_fom_shift_speed(rhs_fitted[:,j], sol_fitted_slice_dx)
        shift_speed_numer[j] = fom.evaluate_fom_shift_speed_numer(rhs_fitted[:,j])
        shift_speed_denom[j] = fom.evaluate_fom_shift_speed_denom(sol_fitted_slice_dx)
    weight_sol = np.mean(np.linalg.norm(sol,axis=0)**2)
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
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,tsave,sol.T)
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    
    if sol_idx == 0:
        plt.title(f"FOM solution, initial condition = uIC")
    elif 1 <= sol_idx <= (n_sol - 1) // 2:
        plt.title(f"FOM solution, initial condition = uIC + {amp} * cos({sol_idx} * x)")
    else:
        plt.title(f"FOM solution, initial condition = uIC + {amp} * sin({sol_idx - (n_sol-1)//2} * x)")

    plt.savefig(fig_path + "sol_FOM_%03d.png"%sol_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,tsave,sol_fitted.T)
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    if sol_idx == 0:
        plt.title(f"FOM solution (fitted), initial condition = uIC")
    elif 1 <= sol_idx <= (n_sol - 1) // 2:
        plt.title(f"FOM solution (fitted), initial condition = uIC + {amp} * cos({sol_idx} * x)")
    else:
        plt.title(f"FOM solution (fitted), initial condition = uIC + {amp} * sin({sol_idx - (n_sol-1)//2} * x)")

    plt.savefig(fig_path + "sol_FOM_fitted_%03d.png"%sol_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(tsave,shift_amount)
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift amount $c(t)$")
    plt.tight_layout()

    if sol_idx == 0:
        plt.title(f"Shift amount, initial condition = uIC")
    elif 1 <= sol_idx <= (n_sol - 1) // 2:
        plt.title(f"Shift amount, initial condition = uIC + {amp} * cos({sol_idx} * x)")
    else:
        plt.title(f"Shift amount, initial condition = uIC + {amp} * sin({sol_idx - (n_sol-1)//2} * x)")
    
    plt.savefig(fig_path + "shift_amount_FOM_%03d.png"%sol_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(tsave,shift_speed_numer, label="numerator")
    plt.plot(tsave,shift_speed_denom, label="denominator")
    plt.plot(tsave,shift_speed, label="shift speed")
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift speed $c'(t)$")
    plt.legend()
    plt.tight_layout()

    if sol_idx == 0:
        plt.title(f"Shift speed, initial condition = uIC")
    elif 1 <= sol_idx <= (n_sol - 1) // 2:
        plt.title(f"Shift speed, initial condition = uIC + {amp} * cos({sol_idx} * x)")
    else:
        plt.title(f"Shift speed, initial condition = uIC + {amp} * sin({sol_idx - (n_sol-1)//2} * x)")
    
    plt.savefig(fig_path + "shift_speed_FOM_%03d.png"%sol_idx)
    plt.close()
    
    ## I want to plot the Fourier components of the solution with time
    ## Plot the Fourier components of the solution with time
    # Perform Fourier transform along the spatial axis for all time steps
    sol_fft_time = np.fft.fft(sol, axis=0)
    
    # Calculate the corresponding integer wavenumbers k
    # For a domain of length L, the wavenumbers are k = n * (2*pi/L). Here L=2*pi, so k=n.
    wavenumbers = np.fft.fftfreq(nx, d=dx) * L

    # 使用 np.fft.fftshift 将频率和傅里叶系数重新排序，
    # 从 [0, 1, ..., N/2-1, -N/2, ..., -1] 变为 [-N/2, ..., -1, 0, 1, ..., N/2-1]
    # This reorders the arrays from the standard FFT output to a centered order.
    wavenumbers_shifted = np.fft.fftshift(wavenumbers)
    sol_fft_time_shifted = np.fft.fftshift(sol_fft_time, axes=0)
    plt.figure(figsize=(10,6))
    plt.pcolormesh(wavenumbers_shifted, tsave, np.abs(sol_fft_time_shifted).T, shading='auto', vmin=0)
    # plt.contourf(wavenumbers_shifted, tsave, np.abs(sol_fft_time_shifted.T))
    plt.colorbar()
    plt.xlabel(r"wavenumber $\xi$")
    plt.ylabel(r"time $t$")

    if sol_idx == 0:
        plt.title(f"Fourier components of the solution, initial condition = uIC")
    elif 1 <= sol_idx <= (n_sol - 1) // 2:
        plt.title(f"Fourier components of the solution, initial condition = uIC + {amp} * cos({sol_idx} * x)")
    else:
        plt.title(f"Fourier components of the solution, initial condition = uIC + {amp} * sin({sol_idx - (n_sol-1)//2} * x)")
    
    plt.savefig(fig_path + "Fourier_components_FOM_%03d.png"%sol_idx)
    plt.close()

    # print the final shift amount
    print("Final shift amount: %.4f"%(shift_amount[-1]))
    
    # print the L2 norm of the initial condition
    print(f"rank = {pool.rank}, L2 norm of the initial condition: {np.linalg.norm(sol_IC_array[:, sol_idx])}")

np.save(sol_path + "time.npy",tsave)

