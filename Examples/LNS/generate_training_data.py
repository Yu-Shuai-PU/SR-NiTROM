import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys
import os
import time as timer
# plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
# plt.rc('text.latex',preamble=r'\usepackage{amsmath}')
sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

import classes
import fom_class_LNS
from fom_class_LNS import Chebyshev_diff_mat as diff_y_mat
from fom_class_LNS import Chebyshev_diff_FFT as diff_y
import nitrom_functions 
import opinf_functions as opinf_fun
import troop_functions
from func_plot import plot_ROM_vs_FOM

"""
Generate the snapshots of 3D linearized NS equations for channel flow

See Henningson, Lundbladh and Johannsson 1993, "A mechanism for bypass transition
from localized disturbances in wall-bounded shear flows" for initial conditions and numerical setups

See Schmid and Henningson 2001, "Stability and Transition in Shear Flows":
    a) p.147 and p.148 Fig. 4.19 for the time evolution (and downstream drifting) of initial localized disturbance.
    b) p.144 Fig. 4.16(c)(f) for the shape of initial localized disturbance.
    c) Fig.1 and eq.1(a-c) and eq.2 from Henningson et al. (1993) for the mathematical form of the initial localized disturbance
    d) Table 1 from Henningson et al. (1993) for the domain size, grid points number, and other parameters used in the simulation
    
Current progress:

1. Successfully verify that our initial condition has zero x-z mean streamwise and spanwise velocity component (u and w) to ensure (u_ff)_{0, 0} = (w_ff)_{0, 0} = 0 (thus the disturbance kinetic energy can be well defined using only v and eta components).
2. Successfully verify that the disturbance kinetic energy computed using (u, v, w) and (v, eta) are consistent, which goes for 1/(2LxLz) int int int u^2 + v^2 + w^2 dy dx dz
3. Successfully verify that the two ways of computing the denominator <u_3D, u_3D_tmp>^2 + <u_3D, T_L/4 u_3D_tmp>^2 are consistent.
4. Successfully verify that the two ways of computing the shifting speed are consistent.

"""

# region 1: Initialization

traj_path = "./trajectories/"
data_path = "./data/"
fig_path  = "./figures/"
os.makedirs(traj_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)

#%% # Generate and save trajectory
fname_traj_template = data_path + "traj_template.npy"
fname_traj_template_dx = data_path + "traj_template_dx.npy" # q_template_dx
fname_traj_template_dx_weighted = data_path + "traj_template_dx_weighted.npy" # R q_template_dx, where W = R^T R is the inner product weight matrix
fname_traj_template_dxx = data_path + "traj_template_dxx.npy"
fname_traj_template_dxx_weighted = data_path + "traj_template_dxx_weighted.npy"
fname_traj_init = data_path + "traj_init_%03d.npy" # for initial condition of u
fname_traj_init_weighted = data_path + "traj_init_weighted_%03d.npy" # for initial condition of u
fname_traj_init_fitted = data_path + "traj_init_fitted_%03d.npy" # for initial condition of u fitted
fname_traj_init_fitted_weighted = data_path + "traj_init_fitted_weighted_%03d.npy" # for initial condition of u fitted
fname_traj = traj_path + "traj_%03d.npy" # for u
fname_traj_weighted = traj_path + "traj_weighted_%03d.npy" # for u
fname_traj_fitted = traj_path + "traj_fitted_%03d.npy" # for u fitted
fname_traj_fitted_weighted = traj_path + "traj_fitted_weighted_%03d.npy" # for u fitted
fname_deriv = traj_path + "deriv_%03d.npy" # for du/dt
fname_deriv_weighted = traj_path + "deriv_weighted_%03d.npy" # for du/dt
fname_deriv_fitted = traj_path + "deriv_fitted_%03d.npy" # for du/dt fitted
fname_deriv_fitted_weighted = traj_path + "deriv_fitted_weighted_%03d.npy" # for du/dt fitted
fname_shift_amount = traj_path + "shift_amount_%03d.npy" # for shifting amount
fname_shift_speed = traj_path + "shift_speed_%03d.npy" # for shifting speed
fname_time = traj_path + "time.npy"

### First, we try to set up the initial localized disturbance

Lx = 32
Ly = 2 # from -1 to 1
Lz = 16

nx = 32
ny = 33 # ny includes the boundary points when using Chebyshev grid
nz = 32

x = np.linspace(0, Lx, num=nx, endpoint=False)
y = np.cos(np.pi * np.linspace(0, ny - 1, num=ny) / (ny - 1))  # Chebyshev grid in y direction, location from 1 to -1
z = np.linspace(0, Lz, num=nz, endpoint=False)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

Re = 3000
# Define the base flow
U_base = 1 - y**2
U_base_dy = -2 * y
U_base_dyy = -2 * np.ones_like(y)

T = 200
dt = 0.5
nsave = 2 # sample interval
time = dt * np.linspace(0, int(T/dt), int(T/dt) + 1, endpoint=True)
tsave = time[::nsave]

fom = fom_class_LNS.LNS(Lx, Ly, Lz, nx, ny, nz, y, Re, U_base, U_base_dy, U_base_dyy)

traj_template = np.load(fname_traj_template)
traj_template_dx = np.load(fname_traj_template_dx)
traj_template_dx_weighted = fom.apply_sqrt_inner_product_weight(traj_template_dx)
np.save(fname_traj_template_dx_weighted, traj_template_dx_weighted)
traj_template_dxx = np.load(fname_traj_template_dxx)
traj_template_dxx_weighted = fom.apply_sqrt_inner_product_weight(traj_template_dxx)
np.save(fname_traj_template_dxx_weighted, traj_template_dxx_weighted)

fom.load_template(traj_template, traj_template_dx)
time = dt * np.linspace(0, int(T/dt), int(T/dt) + 1, endpoint=True)
tstep_kse_fom = fom_class_LNS.time_step_LNS(fom, time)

small_amp  = 0.0001
medium_amp = 0.0699
large_amp  = 0.1398

n_traj = 1

amp = 1.0

psi0 = np.zeros((nx, ny, nz, n_traj))
v0   = np.zeros((nx, ny, nz, n_traj))
eta0 = np.zeros((nx, ny, nz, n_traj))
u0   = np.zeros((nx, ny, nz, n_traj))
w0   = np.zeros((nx, ny, nz, n_traj))

psi0[:, :, :, 0] = amp * (1-Y**2)**2 * ((X - Lx/2)/2) * (Z - Lz/2) ** 2 * np.exp(-((X - Lx/2)/2)**2 - ((Z - Lz/2)/2)**2)
for k in range (n_traj):
    v0[:, :, :, k] = fom.diff_z(psi0[:, :, :, k], order = 1)
    eta0[:, :, :, k] = fom.diff_x(fom.diff_1_y(psi0[:, :, :, k]), order = 1)
    w0[:, :, :, k] = -fom.diff_1_y(psi0[:, :, :, k])
    
# endregion

# region 2: Simulations
pool_inputs = (MPI.COMM_WORLD, n_traj)
pool = classes.mpi_pool(*pool_inputs)

for k in range (pool.my_n_traj):
    traj_idx = k + pool.disps[pool.rank]
    print("Running simulation %d/%d"%(traj_idx,n_traj - 1))
    traj_init = np.concatenate((v0[:, :, :, traj_idx].ravel(), eta0[:, :, :, traj_idx].ravel()))
    
    traj, tsave = tstep_kse_fom.time_step(traj_init, nsave) # traj is in the physical domain and is of the shape (2 * nx * ny * nz, nsave_samples)
    traj_fitted, shifting_amount = fom.template_fitting(traj)
    traj_init_fitted = traj_fitted[:, 0]
    deriv = np.zeros_like(traj)
    deriv_fitted = np.zeros_like(traj_fitted)
    shifting_speed = np.zeros(len(tsave))
    disturbance_energy = np.zeros(len(tsave))
    for i in range (traj.shape[1]):
        deriv[:, i] = fom.evaluate_fom_rhs_unreduced(traj[:, i])
        deriv_v_3D = deriv[:nx * ny * nz, i].reshape((nx, ny, nz))
        deriv_eta_3D = deriv[nx * ny * nz:, i].reshape((nx, ny, nz))
        deriv_v_3D_fitted = fom.shift_x_input_3D(deriv_v_3D, -shifting_amount[i])
        deriv_eta_3D_fitted = fom.shift_x_input_3D(deriv_eta_3D, -shifting_amount[i])
        deriv_fitted[:, i] = np.concatenate((deriv_v_3D_fitted.ravel(), deriv_eta_3D_fitted.ravel()))
        shifting_speed[i] = fom.evaluate_fom_shifting_speed(traj_fitted[:, i], deriv_fitted[:, i])
        
    plt.plot(tsave, shifting_amount)
    plt.xlabel("Time")
    plt.ylabel("Shifting amount c(t)")
    plt.title("Shifting amount over time")
    plt.tight_layout()
    plt.show()
    
    plt.plot(tsave, shifting_speed)
    plt.xlabel("Time")
    plt.ylabel("Shifting speed c'(t)")
    plt.title("Shifting speed over time")
    plt.tight_layout()
    plt.show()
        
    np.save(fname_time, tsave)
    np.save(fname_traj_init%traj_idx,traj_init)
    np.save(fname_traj_init_weighted%traj_idx,fom.apply_sqrt_inner_product_weight(traj_init))
    np.save(fname_traj_init_fitted%traj_idx,traj_init_fitted)
    np.save(fname_traj_init_fitted_weighted%traj_idx,fom.apply_sqrt_inner_product_weight(traj_init_fitted))
    np.save(fname_traj%traj_idx, traj)
    np.save(fname_traj_weighted%traj_idx, fom.apply_sqrt_inner_product_weight(traj))
    np.save(fname_traj_fitted%traj_idx, traj_fitted)
    np.save(fname_traj_fitted_weighted%traj_idx, fom.apply_sqrt_inner_product_weight(traj_fitted))
    np.save(fname_deriv%traj_idx, deriv)
    np.save(fname_deriv_weighted%traj_idx, fom.apply_sqrt_inner_product_weight(deriv))
    np.save(fname_deriv_fitted%traj_idx, deriv_fitted)
    np.save(fname_deriv_fitted_weighted%traj_idx, fom.apply_sqrt_inner_product_weight(deriv_fitted))
    np.save(fname_shift_amount%traj_idx, shifting_amount)
    np.save(fname_shift_speed%traj_idx, shifting_speed)
    
    traj_v = traj[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
    traj_eta = traj[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
    traj_v_fitted = traj_fitted[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
    traj_eta_fitted = traj_fitted[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
    deriv_v = deriv[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
    deriv_eta = deriv[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
    deriv_v_fitted = deriv_fitted[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
    deriv_eta_fitted = deriv_fitted[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
    
    t_check_list = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]
    y_check = -0.56
    
    v_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
    eta_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
    v_fitted_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
    eta_fitted_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
    deriv_v_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
    deriv_eta_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
    deriv_v_fitted_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
    deriv_eta_fitted_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))

    for t_check in t_check_list:

        idx_sample = int(t_check / (dt * nsave))
        v_slice = traj_v[:, :, :, idx_sample]
        eta_slice = traj_eta[:, :, :, idx_sample]
        v_fitted_slice = traj_v_fitted[:, :, :, idx_sample]
        eta_fitted_slice = traj_eta_fitted[:, :, :, idx_sample]
        deriv_v_slice = deriv_v[:, :, :, idx_sample]
        deriv_eta_slice = deriv_eta[:, :, :, idx_sample]
        deriv_v_fitted_slice = deriv_v_fitted[:, :, :, idx_sample]
        deriv_eta_fitted_slice = deriv_eta_fitted[:, :, :, idx_sample]
        idx_y_check = np.argmin(np.abs(y - y_check))
        v_slice_ycheck = v_slice[:, idx_y_check, :]
        v_fitted_slice_ycheck = v_fitted_slice[:, idx_y_check, :]
        eta_slice_ycheck = eta_slice[:, idx_y_check, :]
        eta_fitted_slice_ycheck = eta_fitted_slice[:, idx_y_check, :]
        deriv_v_slice_ycheck = deriv_v_slice[:, idx_y_check, :]
        deriv_v_fitted_slice_ycheck = deriv_v_fitted_slice[:, idx_y_check, :]
        deriv_eta_slice_ycheck = deriv_eta_slice[:, idx_y_check, :]
        deriv_eta_fitted_slice_ycheck = deriv_eta_fitted_slice[:, idx_y_check, :]
        
        v_fitted_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = v_fitted_slice_ycheck[:, nz//2]
        eta_fitted_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = eta_fitted_slice_ycheck[:, nz//2]
        v_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = v_slice_ycheck[:, nz//2]
        eta_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = eta_slice_ycheck[:, nz//2]
        deriv_v_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = deriv_v_slice_ycheck[:, nz//2]
        deriv_eta_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = deriv_eta_slice_ycheck[:, nz//2]
        deriv_v_fitted_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = deriv_v_fitted_slice_ycheck[:, nz//2]
        deriv_eta_fitted_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = deriv_eta_fitted_slice_ycheck[:, nz//2]  

        v_min = np.min(v_slice_ycheck)
        v_max = np.max(v_slice_ycheck)
        # v_spacing = 1e-6  # 等高线间距
        
        eta_min = np.min(eta_slice_ycheck)
        eta_max = np.max(eta_slice_ycheck)
        # eta_spacing = 1e-6  # 等高线间距

        # 构造等高线 levels
        # levels = np.arange(v_min - v_spacing, v_max + v_spacing, v_spacing)
        # plt.figure(figsize=(10,6))
        # # plt.contourf(x, z, v_slice_ycheck.T, levels=levels, cmap='jet')
        # plt.pcolormesh(x, z, v_slice_ycheck.T, cmap='bwr')
        # plt.colorbar()
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$z$")
        # plt.xlim(np.min(x), np.max(x))
        # plt.ylim(np.min(z), np.max(z))
        # plt.title(f"Normal velocity v at t={t_check}, y={y_check}")
        # plt.tight_layout()
        # plt.show()
        
        # plt.figure(figsize=(10, 6))
        # # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
        # cs = plt.contour(x, z, v_slice_ycheck.T, colors='black', linewidths=0.6)
        # # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
        # # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
        # # plt.colorbar()
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$z$")
        # plt.xlim(np.min(x), np.max(x))
        # plt.ylim(np.min(z), np.max(z))
        # plt.title(f"Contours of normal velocity v at t={t_check}, y={y_check}")
        # plt.tight_layout()
        # plt.show()
        
        # plt.figure(figsize=(10,6))
        # # plt.contourf(x, z, v_slice_ycheck.T, levels=levels, cmap='jet')
        # plt.pcolormesh(x, z, v_fitted_slice_ycheck.T, cmap='bwr')
        # plt.colorbar()
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$z$")
        # plt.xlim(np.min(x), np.max(x))
        # plt.ylim(np.min(z), np.max(z))
        # plt.title(f"Fitted normal velocity v at t={t_check}, y={y_check}")
        # plt.tight_layout()
        # plt.show()
        
        # plt.figure(figsize=(10, 6))
        # # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
        # cs = plt.contour(x, z, v_fitted_slice_ycheck.T, colors='black', linewidths=0.6)
        # # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
        # # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
        # # plt.colorbar()
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$z$")
        # plt.xlim(np.min(x), np.max(x))
        # plt.ylim(np.min(z), np.max(z))
        # plt.title(f"Contours of fitted normal velocity v at t={t_check}, y={y_check}")
        # plt.tight_layout()
        # plt.show()
        
    plt.figure(figsize=(10,6))
    for i in range (len(t_check_list)):
        plt.plot(x, eta_fitted_slice_ycheck_all_z_centered[:, i], label=f"t={t_check_list[i]}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"Fitted normal vorticity at y={}".format(y_check))
    plt.title("Fitted normal vorticity at y={} over time".format(y_check))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,6))
    for i in range (len(t_check_list)):
        plt.plot(x, eta_slice_ycheck_all_z_centered[:, i], label=f"t={t_check_list[i]}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"Normal vorticity at y={}".format(y_check))
    plt.title("Normal vorticity at y={} over time".format(y_check))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,6))
    for i in range (len(t_check_list)):
        plt.plot(x, deriv_eta_slice_ycheck_all_z_centered[:, i], label=f"t={t_check_list[i]}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"RHS of the normal vorticity at y={}".format(y_check))
    plt.title("RHS of the normal vorticity at y={} over time".format(y_check))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,6))
    for i in range (len(t_check_list)):
        plt.plot(x, deriv_eta_fitted_slice_ycheck_all_z_centered[:, i], label=f"t={t_check_list[i]}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"Fitted RHS of the normal vorticity at y={}".format(y_check))
    plt.title("Fitted RHS of the normal vorticity at y={} over time".format(y_check))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# endregion





