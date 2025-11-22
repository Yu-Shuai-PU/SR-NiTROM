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

# cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
# lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'

def smooth_bump(x, mu, w):
    # """
    # Create a C-infinity "bump" function centered at mu with width approximately w.
    # It is zero for |x-mu| >= w.
    # """
    arg = (x - mu) / w
    bump = np.zeros_like(x)
    # Find indices where |arg| < 1
    mask = np.abs(arg) < 1.0
    # Only compute the "bump" for these indices
    bump[mask] = np.exp(-1.0 / (1.0 - arg[mask]**2))
    return bump

#%% # Instantiate KSE class and KSE time-stepper class

cmap_name = 'bwr'  # Colormap for contour plots
contourf_vmax = 16
contourf_levels = np.linspace(-contourf_vmax, contourf_vmax, 9)
fontsize = 20
# time_ticks = [0, int(T / 4), int(T / 2), int(3 * T / 4), int(T)]
shift_amount_ticks = [-1, -0.5, 0, 0.5, 1]
shift_speed_ticks = [-4, -2, 0, 2, 4]


traj_path = "./trajectories_testing/"
data_path = "./data/"
fig_path = "./figures_testing/"
os.makedirs(traj_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)

fname_X_template = data_path + "X_template.npy"
fname_X_template_dx = data_path + "X_template_dx.npy"
fname_X_template_dxx = data_path + "X_template_dxx.npy"
fname_time_testing = traj_path + "time_testing.npy"
fname_traj_FOM = traj_path + "traj_FOM_%03d.npy" # for u
fname_traj_fitted_FOM = traj_path + "traj_fitted_FOM_%03d.npy"
fname_shift_amount_FOM = traj_path + "shift_amount_FOM_%03d.npy" # for shifting amount
fname_shift_speed_FOM = traj_path + "shift_speed_FOM_%03d.npy"

L = 2 * np.pi
nx = 40
x = np.linspace(0, L, num=nx, endpoint=False)
nu = 4/87
maximal_unstable_wavenumber = int(np.sqrt(1/nu))

nmodes_ic_perturbation = 2 * maximal_unstable_wavenumber
modes_ic_perturbation  = np.zeros((nx, nmodes_ic_perturbation))
for k in range (1, maximal_unstable_wavenumber + 1):
    modes_ic_perturbation[:, 2 * (k - 1)] = np.cos(k * x)
    modes_ic_perturbation[:, 2 * (k - 1) + 1] = np.sin(k * x)

dx = x[1] - x[0]
dt = 1e-3
T = 5
time = dt * np.linspace(0, int(T/dt), int(T/dt) + 1, endpoint=True)
time_ticks = [0, time[-1]/5, 2 * time[-1]/5, 3*time[-1]/5, 4*time[-1]/5, time[-1]]


nsave = 10
tsave = time[::nsave]
start_time = 80
np.save(fname_time_testing, tsave)

n_traj_testing = 7
pool_inputs = (MPI.COMM_WORLD, n_traj_testing)
pool_kwargs = {'fname_X_template':fname_X_template, 'fname_X_template_dx':fname_X_template_dx,
               'fname_X_template_dxx':fname_X_template_dxx, 'fname_time':fname_time_testing}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)
pool.load_data()
fom = fom_class_kse.KSE(L, nu, nx, pool.X_template, pool.X_template_dx)
tstep_kse_fom = fom_class_kse.time_step_kse(fom, time)

#%% # Generate and save FOM trajectory
poly_comp = [1, 2] # polynomial degree for the ROM dynamics
ic_perturbation_name = "noise"
traj_init_array = np.zeros((nx, n_traj_testing))
traj_init_original = np.loadtxt(f"initial_condition_time_{start_time}.txt").reshape(-1) 
perturbation_amp_proportion = 0.1
amp = perturbation_amp_proportion * np.linalg.norm(traj_init_original)
amp_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) * amp

for idx_k in range(n_traj_testing):
    random_seed = 42 + idx_k
    np.random.seed(random_seed)
    random_noise_raw = np.random.normal(0, 1, size=nmodes_ic_perturbation)
    ic_perturbation = modes_ic_perturbation @ random_noise_raw
    traj_init_array[:, idx_k] = traj_init_original + amp_array[idx_k] * ic_perturbation / np.linalg.norm(ic_perturbation)
    if pool.rank == 0:
        print(f"initial condition No.{idx_k + 1}, perturbation amp proportion = {np.sqrt(np.linalg.norm(traj_init_array[:, idx_k] - traj_init_original)**2 / np.linalg.norm(traj_init_original)**2):.4f}")
np.save(fname_time_testing, tsave)


for k in range (pool.my_n_traj):
    traj_idx = k + pool.disps[pool.rank]
    print("Running simulation %d/%d"%(traj_idx + 1,n_traj_testing))
    X0 = traj_init_array[:,traj_idx]

    X_FOM, tsave = tstep_kse_fom.time_step(X0, nsave)
    X_fitted_FOM, c_FOM = fom.template_fitting(X_FOM, pool.X_template)
    X0_fitted = X_fitted_FOM[:,0]
    cdot_FOM = np.zeros_like(c_FOM)
    dX_FOM = np.zeros_like(X_FOM)
    dX_fitted_FOM = np.zeros_like(X_FOM)
    for j in range (X_FOM.shape[-1]):
        dX_FOM[:, j] = fom.evaluate_fom_rhs(0.0, X_FOM[:,j], np.zeros(X_FOM.shape[0]))
        dX_fitted_FOM[:, j] = fom.shift(dX_FOM[:,j], -c_FOM[j])
        cdot_FOM[j] = fom.evaluate_fom_shift_speed(dX_fitted_FOM[:,j], fom.spatial_deriv(X_fitted_FOM[:,j], order = 1))

    np.save(fname_traj_FOM%traj_idx,X_FOM)
    np.save(fname_traj_fitted_FOM%traj_idx,X_fitted_FOM)
    np.save(fname_shift_amount_FOM%traj_idx,c_FOM)
    np.save(fname_shift_speed_FOM%traj_idx,cdot_FOM)
    
    ### Plotting, things to be done:
    ### 1. switch from contourf to pcolormesh
    ### 2. apply Fourier spectral interpolation to plot a 40-mode solution on a 256-point grid for better visualization
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,tsave,X_FOM.T, levels = contourf_levels, cmap=cmap_name)
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.xlabel("x", fontsize=fontsize)
    plt.yticks(time_ticks, fontsize = fontsize)
    plt.ylabel("t", fontsize=fontsize)
    plt.tight_layout()
    plt.title(f"FOM solution, initial condition = uIC + {amp_array[traj_idx]} * {ic_perturbation_name}")
    plt.savefig(fig_path + "traj_FOM_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,tsave,X_fitted_FOM.T, levels = contourf_levels, cmap=cmap_name)
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.xlabel("x", fontsize=fontsize)
    plt.yticks(time_ticks, fontsize = fontsize)
    plt.ylabel("t", fontsize=fontsize)
    plt.tight_layout()
    plt.title(f"Fitted FOM solution, initial condition = uIC + {amp_array[traj_idx]} * {ic_perturbation_name}")
    plt.savefig(fig_path + "traj_FOM_fitted_%03d.png"%traj_idx)
    plt.close()
    
# endregion