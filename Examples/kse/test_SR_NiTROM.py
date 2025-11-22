import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys
import os

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from pymanopt.tools.diagnostics import check_gradient
# plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
# plt.rc('text.latex',preamble=r'\usepackage{amsmath}')

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
fname_traj_SRG = traj_path + "traj_SRG_%03d.npy" # for u
fname_traj_fitted_SRG = traj_path + "traj_fitted_SRG_%03d.npy"
fname_shift_amount_SRG = traj_path + "shift_amount_SRG_%03d.npy" # for shifting amount
fname_shift_speed_SRG = traj_path + "shift_speed_SRG_%03d.npy"
fname_relative_error_SRG = traj_path + "relative_error_SRG_%03d.npy"
fname_traj_SRN = traj_path + "traj_SRN_%03d.npy" # for u
fname_traj_fitted_SRN = traj_path + "traj_fitted_SRN_%03d.npy"
fname_shift_amount_SRN = traj_path + "shift_amount_SRN_%03d.npy" # for shifting amount
fname_shift_speed_SRN = traj_path + "shift_speed_SRN_%03d.npy"
fname_relative_error_SRN = traj_path + "relative_error_SRN_%03d.npy"

L = 2 * np.pi
nx = 40
x = np.linspace(0, L, num=nx, endpoint=False)
nu = 4/87

n_traj_testing = 7
pool_inputs = (MPI.COMM_WORLD, n_traj_testing)
pool_kwargs = {'fname_time':fname_time_testing, 'fname_traj':fname_traj_FOM,'fname_traj_fitted':fname_traj_fitted_FOM,
               'fname_X_template':fname_X_template, 'fname_X_template_dx':fname_X_template_dx,
               'fname_X_template_dxx':fname_X_template_dxx,
               'fname_shift_amount':fname_shift_amount_FOM,'fname_shift_speed':fname_shift_speed_FOM}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)
pool.load_data()
T = pool.time[-1]
fom = fom_class_kse.KSE(L, nu, nx, pool.X_template, pool.X_template_dx)

#%% # Generate and save FOM trajectory
start_time = 80
poly_comp = [1, 2] # polynomial degree for the ROM dynamics
ic_perturbation_name = "noise"
traj_init_array = np.zeros((nx, n_traj_testing))
traj_init_original = np.loadtxt(f"initial_condition_time_{start_time}.txt").reshape(-1) 
perturbation_amp_proportion = 0.1
amp = perturbation_amp_proportion * np.linalg.norm(traj_init_original)
amp_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) * amp

#%% # Load trained bases and tensors (SRG and SRN)

Phi_POD = np.load(data_path + "Phi_POD.npy")
Psi_POD = np.load(data_path + "Psi_POD.npy")
PhiF_POD = Phi_POD@scipy.linalg.inv(Psi_POD.T@Phi_POD)
Tensors_POD_Galerkin_file = np.load(data_path + "Tensors_POD_Galerkin.npz")
Tensors_POD = (
    Tensors_POD_Galerkin_file['arr_0'],
    Tensors_POD_Galerkin_file['arr_1'],
    Tensors_POD_Galerkin_file['arr_2'],
    Tensors_POD_Galerkin_file['arr_3'],
    Tensors_POD_Galerkin_file['arr_4'],
    Tensors_POD_Galerkin_file['arr_5'])

r = Phi_POD.shape[1] # ROM dimension

Phi_NiTROM = np.load(data_path + "Phi_NiTROM.npy")
Psi_NiTROM = np.load(data_path + "Psi_NiTROM.npy")
PhiF_NiTROM = Phi_NiTROM @ scipy.linalg.inv(Psi_NiTROM.T@Phi_NiTROM)
Tensors_NiTROM_file = np.load(data_path + "Tensors_NiTROM.npz")
Tensors_NiTROM = (
    Tensors_NiTROM_file['arr_0'],
    Tensors_NiTROM_file['arr_1'],
    Tensors_NiTROM_file['arr_2'],
    Tensors_NiTROM_file['arr_3'],
    Tensors_NiTROM_file['arr_4'],
    Tensors_NiTROM_file['arr_5'])

which_trajs = np.arange(0, pool.my_n_traj, 1)
which_times = np.arange(0, pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 11

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj_kwargs = {
    'X_template_dx': pool.X_template_dx,
    'X_template_dxx': pool.X_template_dxx,
    'spatial_deriv_method': fom.spatial_deriv,
    'inner_product_method': fom.inner_product,
    'outer_product_method': fom.outer_product}
opt_obj = classes.optimization_objects(*opt_obj_inputs, **opt_obj_kwargs)

## FOM and ROMs simulation

relative_error_SRG = np.zeros(pool.n_snapshots)
relative_error_space_time_SRG = np.zeros(n_traj_testing)
relative_error_SRN = np.zeros(pool.n_snapshots)
relative_error_space_time_SRN = np.zeros(n_traj_testing)

time_ticks = [0, int(T / 5), 2 * int(T / 5), 3 * int(T / 5), 4 * int(T / 5), int(T)]

for k in range (pool.my_n_traj):
    traj_idx = k + pool.disps[pool.rank]
    print("Running simulation %d/%d"%(traj_idx + 1,n_traj_testing))
    
    z0 = Psi_POD.T@opt_obj.X_fitted[k,:,0].reshape(-1)
    c0 = opt_obj.c[k,0]

    sol = solve_ivp(opt_obj.evaluate_rom_rhs,
                    [opt_obj.time[0],opt_obj.time[-1]],
                    np.hstack((z0, c0)),
                    'RK45',
                    t_eval=opt_obj.time,
                    args=(np.zeros(r),) + Tensors_POD).y
    
    X_fitted_SRG = PhiF_POD@sol[:-1,:]
    c_SRG = sol[-1,:]
    X_SRG = np.zeros_like(X_fitted_SRG)
    cdot_SRG = np.zeros_like(c_SRG)
    for j in range (len(opt_obj.time)):
        X_SRG[:,j] = fom.shift(X_fitted_SRG[:,j], c_SRG[j])
        cdot_SRG[j] = opt_obj.compute_shift_speed(sol[:r,j], Tensors_POD)
        relative_error_SRG[j] = np.linalg.norm(opt_obj.X[k,:,j] - X_SRG[:,j]) / np.linalg.norm(opt_obj.X[k,:,j])
        
    relative_error_space_time_SRG[traj_idx] = np.linalg.norm(opt_obj.X[k,:,:] - X_SRG)/np.linalg.norm(opt_obj.X[k,:,:])
        
    np.save(fname_traj_SRG%traj_idx,X_SRG)
    np.save(fname_traj_fitted_SRG%traj_idx,X_fitted_SRG)
    np.save(fname_shift_amount_SRG%traj_idx,c_SRG)
    np.save(fname_shift_speed_SRG%traj_idx,cdot_SRG)
    np.save(fname_relative_error_SRG%traj_idx,relative_error_SRG)
    
    z0 = Psi_NiTROM.T@opt_obj.X_fitted[k,:,0].reshape(-1)
    c0 = opt_obj.c[k,0]

    sol = solve_ivp(opt_obj.evaluate_rom_rhs,
                    [opt_obj.time[0],opt_obj.time[-1]],
                    np.hstack((z0, c0)),
                    'RK45',
                    t_eval=opt_obj.time,
                    args=(np.zeros(r),) + Tensors_NiTROM).y
    
    X_fitted_SRN = PhiF_NiTROM@sol[:-1,:]
    c_SRN = sol[-1,:]
    X_SRN = np.zeros_like(X_fitted_SRN)
    cdot_SRN = np.zeros_like(c_SRN)
    for j in range (len(opt_obj.time)):
        X_SRN[:,j] = fom.shift(X_fitted_SRN[:,j], c_SRN[j])
        cdot_SRN[j] = opt_obj.compute_shift_speed(sol[:r,j], Tensors_NiTROM)
        relative_error_SRN[j] = np.linalg.norm(opt_obj.X[k,:,j] - X_SRN[:,j]) / np.linalg.norm(opt_obj.X[k,:,j])
        
    relative_error_space_time_SRN[traj_idx] = np.linalg.norm(opt_obj.X[k,:,:] - X_SRN)/np.linalg.norm(opt_obj.X[k,:,:])
        
    np.save(fname_traj_SRN%traj_idx,X_SRN)
    np.save(fname_traj_fitted_SRN%traj_idx,X_fitted_SRN)
    np.save(fname_shift_amount_SRN%traj_idx,c_SRN)
    np.save(fname_shift_speed_SRN%traj_idx,cdot_SRN)
    np.save(fname_relative_error_SRN%traj_idx,relative_error_SRN)
    
    ### Plotting, things to be done:
    ### 1. switch from contourf to pcolormesh
    ### 2. apply Fourier spectral interpolation to plot a 40-mode solution on a 256-point grid for better visualization
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,X_SRG.T, levels = contourf_levels, cmap=cmap_name)
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.xlabel("x", fontsize=fontsize)
    plt.yticks(time_ticks, fontsize = fontsize)
    plt.ylabel("t", fontsize=fontsize)
    plt.tight_layout()
    plt.title(f"SR-Galerkin solution, initial condition = uIC + {amp_array[traj_idx]} * {ic_perturbation_name}, error = {relative_error_space_time_SRG[traj_idx]:.4e}")
    plt.savefig(fig_path + "traj_SRG_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,X_fitted_SRG.T, levels = contourf_levels, cmap=cmap_name)
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.xlabel("x", fontsize=fontsize)
    plt.yticks(time_ticks, fontsize = fontsize)
    plt.ylabel("t", fontsize=fontsize)
    plt.tight_layout()
    plt.title(f"Fitted SR-Galerkin solution, initial condition = uIC + {amp_array[traj_idx]} * {ic_perturbation_name}")
    plt.savefig(fig_path + "traj_SRG_fitted_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,X_SRN.T, levels = contourf_levels, cmap=cmap_name)
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.xlabel("x", fontsize=fontsize)
    plt.yticks(time_ticks, fontsize = fontsize)
    plt.ylabel("t", fontsize=fontsize)
    plt.tight_layout()
    plt.title(f"SR-NiTROM solution, initial condition = uIC + {amp_array[traj_idx]} * {ic_perturbation_name}, error = {relative_error_space_time_SRN[traj_idx]:.4e}")
    plt.savefig(fig_path + "traj_SRN_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,X_fitted_SRN.T, levels = contourf_levels, cmap=cmap_name)
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.xlabel("x", fontsize=fontsize)
    plt.yticks(time_ticks, fontsize = fontsize)
    plt.ylabel("t", fontsize=fontsize)
    plt.tight_layout()
    plt.title(f"Fitted SR-NiTROM solution, initial condition = uIC + {amp_array[traj_idx]} * {ic_perturbation_name}")
    plt.savefig(fig_path + "traj_SRN_fitted_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time,opt_obj.c[k,:], label = "FOM")
    plt.plot(opt_obj.time,c_SRG, label = "SR-Galerkin")
    plt.plot(opt_obj.time,c_SRN, label = "SR-NiTROM")
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift amount $c(t)$")
    plt.tight_layout()
    plt.title(f"Shift amount, initial condition = uIC + {amp_array[traj_idx]} * {ic_perturbation_name}")
    
    plt.savefig(fig_path + "shift_amount_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time,opt_obj.cdot[k,:], label="FOM")
    plt.plot(opt_obj.time,cdot_SRG, label="SR-Galerkin")
    plt.plot(opt_obj.time,cdot_SRN, label="SR-NiTROM")
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift speed $c'(t)$")
    plt.legend()
    plt.tight_layout()

    plt.title(f"Shift speed, initial condition = uIC + {amp_array[traj_idx]} * {ic_perturbation_name}")
    
    plt.savefig(fig_path + "shift_speed_%03d.png"%traj_idx)
    plt.close()
   
relative_error_space_time_SRG = np.sum(np.asarray(pool.comm.allgather(relative_error_space_time_SRG)), axis=0)
relative_error_space_time_SRN = np.sum(np.asarray(pool.comm.allgather(relative_error_space_time_SRN)), axis=0)
   
if pool.rank == 0:
    print("Mean relative error of SR-Galerkin for all solutions: %.4e"%(np.mean(relative_error_space_time_SRG)))
    print("Mean relative error of SR-NiTROM for all solutions: %.4e"%(np.mean(relative_error_space_time_SRN)))

# endregion
