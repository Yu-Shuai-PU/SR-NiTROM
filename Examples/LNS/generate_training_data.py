import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy.integrate import cumulative_trapezoid


from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys
import os
import time as timer
# plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
# plt.rc('text.latex',preamble=r'\usepackage{amsmath}')
sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

import configs
import classes
import fom_class_LNS
import nitrom_functions 
import opinf_functions as opinf_fun

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
params = configs.load_configs()
fom = fom_class_LNS.LNS(params.Lx, params.Ly, params.Lz, 
                        params.nx, params.ny, params.nz,
                        params.y, params.Re,
                        params.U_base, params.U_base_dy, params.U_base_dyy)
tstep_kse_fom = fom_class_LNS.time_step_LNS(fom, params.time)

# Load trajectory template and related information
traj_template = np.load(params.fname_traj_template)
traj_template_dx = np.load(params.fname_traj_template_dx)
traj_template_dx_weighted = fom.apply_sqrt_inner_product_weight(traj_template_dx)
np.save(params.fname_traj_template_dx_weighted, traj_template_dx_weighted)
traj_template_dxx = np.load(params.fname_traj_template_dxx)
traj_template_dxx_weighted = fom.apply_sqrt_inner_product_weight(traj_template_dxx)
np.save(params.fname_traj_template_dxx_weighted, traj_template_dxx_weighted)
fom.load_template(traj_template, traj_template_dx)

v0   = np.zeros((params.nx, params.ny, params.nz, params.n_traj))
eta0 = np.zeros((params.nx, params.ny, params.nz, params.n_traj))

for idx in range (params.n_traj):
    v0[:, :, :, idx] = fom.diff_z(params.psi0[:, :, :, idx], order = 1)
    eta0[:, :, :, idx] = fom.diff_x(fom.diff_1_y(params.psi0[:, :, :, idx]), order = 1)
    
# endregion

# region 2: Simulations
pool_inputs = (MPI.COMM_WORLD, params.n_traj)
pool = classes.mpi_pool(*pool_inputs)


for k in range (pool.my_n_traj):
    traj_idx = k + pool.disps[pool.rank]
    print("Running simulation %d/%d"%(traj_idx,params.n_traj - 1))
    traj_init = np.concatenate((v0[:, :, :, traj_idx].ravel(), eta0[:, :, :, traj_idx].ravel()))
    
    traj, tsave = tstep_kse_fom.time_step(traj_init, params.nsave) # traj is in the physical domain and is of the shape (2 * nx * ny * nz, nsave_samples)
    traj_fitted, shifting_amount = fom.template_fitting(traj)
    traj_init_fitted = traj_fitted[:, 0]
    deriv = np.zeros_like(traj)
    deriv_fitted = np.zeros_like(traj_fitted)
    shifting_speed = np.zeros(len(tsave))
    disturbance_energy = np.zeros(len(tsave))
    for i in range (traj.shape[1]):
        deriv[:, i] = fom.evaluate_fom_rhs_unreduced(traj[:, i])
        deriv_v_3D = deriv[:params.nx * params.ny * params.nz, i].reshape((params.nx, params.ny, params.nz))
        deriv_eta_3D = deriv[params.nx * params.ny * params.nz:, i].reshape((params.nx, params.ny, params.nz))
        deriv_v_3D_fitted = fom.shift_x_input_3D(deriv_v_3D, -shifting_amount[i])
        deriv_eta_3D_fitted = fom.shift_x_input_3D(deriv_eta_3D, -shifting_amount[i])
        deriv_fitted[:, i] = np.concatenate((deriv_v_3D_fitted.ravel(), deriv_eta_3D_fitted.ravel()))
        shifting_speed[i] = fom.evaluate_fom_shifting_speed(traj_fitted[:, i], deriv_fitted[:, i])
        disturbance_energy[i] = fom.inner_product_3D(traj[:params.nx * params.ny * params.nz, i].reshape((params.nx, params.ny, params.nz)),
                                                     traj[params.nx * params.ny * params.nz:, i].reshape((params.nx, params.ny, params.nz)),
                                                     traj[:params.nx * params.ny * params.nz, i].reshape((params.nx, params.ny, params.nz)),
                                                     traj[params.nx * params.ny * params.nz:, i].reshape((params.nx, params.ny, params.nz)))        
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
    
    plt.plot(tsave, disturbance_energy)
    plt.xlabel("Time")
    plt.ylabel("Disturbance Kinetic Energy")
    plt.title("Disturbance Kinetic Energy over time")
    plt.tight_layout()
    plt.show()
        
    np.save(params.fname_time, tsave)
    np.save(params.fname_traj_init%traj_idx,traj_init)
    np.save(params.fname_traj_init_weighted%traj_idx,fom.apply_sqrt_inner_product_weight(traj_init))
    np.save(params.fname_traj_init_fitted%traj_idx,traj_init_fitted)
    np.save(params.fname_traj_init_fitted_weighted%traj_idx,fom.apply_sqrt_inner_product_weight(traj_init_fitted))
    np.save(params.fname_traj%traj_idx, traj)
    np.save(params.fname_traj_weighted%traj_idx, fom.apply_sqrt_inner_product_weight(traj))
    np.save(params.fname_traj_fitted%traj_idx, traj_fitted)
    np.save(params.fname_traj_fitted_weighted%traj_idx, fom.apply_sqrt_inner_product_weight(traj_fitted))
    np.save(params.fname_deriv%traj_idx, deriv)
    np.save(params.fname_deriv_weighted%traj_idx, fom.apply_sqrt_inner_product_weight(deriv))
    np.save(params.fname_deriv_fitted%traj_idx, deriv_fitted)
    np.save(params.fname_deriv_fitted_weighted%traj_idx, fom.apply_sqrt_inner_product_weight(deriv_fitted))
    np.save(params.fname_shift_amount%traj_idx, shifting_amount)
    np.save(params.fname_shift_speed%traj_idx, shifting_speed)
    
    traj_v = traj[0 : params.nx * params.ny * params.nz, :].reshape((params.nx, params.ny, params.nz, -1))
    traj_eta = traj[params.nx * params.ny * params.nz : , :].reshape((params.nx, params.ny, params.nz, -1))
    traj_v_fitted = traj_fitted[0 : params.nx * params.ny * params.nz, :].reshape((params.nx, params.ny, params.nz, -1))
    traj_eta_fitted = traj_fitted[params.nx * params.ny * params.nz : , :].reshape((params.nx, params.ny, params.nz, -1))
    deriv_v = deriv[0 : params.nx * params.ny * params.nz, :].reshape((params.nx, params.ny, params.nz, -1))
    deriv_eta = deriv[params.nx * params.ny * params.nz : , :].reshape((params.nx, params.ny, params.nz, -1))
    deriv_v_fitted = deriv_fitted[0 : params.nx * params.ny * params.nz, :].reshape((params.nx, params.ny, params.nz, -1))
    deriv_eta_fitted = deriv_fitted[params.nx * params.ny * params.nz : , :].reshape((params.nx, params.ny, params.nz, -1))
    
    v_slice_ycheck_all_z_centered = np.zeros((params.nx, len(params.t_check_list_POD)))
    eta_slice_ycheck_all_z_centered = np.zeros((params.nx, len(params.t_check_list_POD)))
    v_fitted_slice_ycheck_all_z_centered = np.zeros((params.nx, len(params.t_check_list_POD)))
    eta_fitted_slice_ycheck_all_z_centered = np.zeros((params.nx, len(params.t_check_list_POD)))
    deriv_v_slice_ycheck_all_z_centered = np.zeros((params.nx, len(params.t_check_list_POD)))
    deriv_eta_slice_ycheck_all_z_centered = np.zeros((params.nx, len(params.t_check_list_POD)))
    deriv_v_fitted_slice_ycheck_all_z_centered = np.zeros((params.nx, len(params.t_check_list_POD)))
    deriv_eta_fitted_slice_ycheck_all_z_centered = np.zeros((params.nx, len(params.t_check_list_POD)))
    for t_check in params.t_check_list_POD:

        idx_sample = int(t_check / (params.dt * params.nsave))
        v_slice = traj_v[:, :, :, idx_sample]
        eta_slice = traj_eta[:, :, :, idx_sample]
        v_fitted_slice = traj_v_fitted[:, :, :, idx_sample]
        eta_fitted_slice = traj_eta_fitted[:, :, :, idx_sample]
        deriv_v_slice = deriv_v[:, :, :, idx_sample]
        deriv_eta_slice = deriv_eta[:, :, :, idx_sample]
        deriv_v_fitted_slice = deriv_v_fitted[:, :, :, idx_sample]
        deriv_eta_fitted_slice = deriv_eta_fitted[:, :, :, idx_sample]
        idx_y_check = np.argmin(np.abs(params.y - params.y_check))
        v_slice_ycheck = v_slice[:, idx_y_check, :]
        v_fitted_slice_ycheck = v_fitted_slice[:, idx_y_check, :]
        eta_slice_ycheck = eta_slice[:, idx_y_check, :]
        eta_fitted_slice_ycheck = eta_fitted_slice[:, idx_y_check, :]
        deriv_v_slice_ycheck = deriv_v_slice[:, idx_y_check, :]
        deriv_v_fitted_slice_ycheck = deriv_v_fitted_slice[:, idx_y_check, :]
        deriv_eta_slice_ycheck = deriv_eta_slice[:, idx_y_check, :]
        deriv_eta_fitted_slice_ycheck = deriv_eta_fitted_slice[:, idx_y_check, :]
        
        v_fitted_slice_ycheck_all_z_centered[:, params.t_check_list_POD.index(t_check)] = v_fitted_slice_ycheck[:, params.nz//2]
        eta_fitted_slice_ycheck_all_z_centered[:, params.t_check_list_POD.index(t_check)] = eta_fitted_slice_ycheck[:, params.nz//2]
        v_slice_ycheck_all_z_centered[:, params.t_check_list_POD.index(t_check)] = v_slice_ycheck[:, params.nz//2]
        eta_slice_ycheck_all_z_centered[:, params.t_check_list_POD.index(t_check)] = eta_slice_ycheck[:, params.nz//2]
        deriv_v_slice_ycheck_all_z_centered[:, params.t_check_list_POD.index(t_check)] = deriv_v_slice_ycheck[:, params.nz//2]
        deriv_eta_slice_ycheck_all_z_centered[:, params.t_check_list_POD.index(t_check)] = deriv_eta_slice_ycheck[:, params.nz//2]
        deriv_v_fitted_slice_ycheck_all_z_centered[:, params.t_check_list_POD.index(t_check)] = deriv_v_fitted_slice_ycheck[:, params.nz//2]
        deriv_eta_fitted_slice_ycheck_all_z_centered[:, params.t_check_list_POD.index(t_check)] = deriv_eta_fitted_slice_ycheck[:, params.nz//2]  

        # v_min = np.min(v_slice_ycheck)
        # v_max = np.max(v_slice_ycheck)
        # # v_spacing = 1e-6  # 等高线间距
        
        # eta_min = np.min(eta_slice_ycheck)
        # eta_max = np.max(eta_slice_ycheck)
        # # eta_spacing = 1e-6  # 等高线间距

        # # 构造等高线 levels
        # # levels = np.arange(v_min - v_spacing, v_max + v_spacing, v_spacing)
        # plt.figure(figsize=(10,6))
        # # plt.contourf(x, z, v_slice_ycheck.T, levels=levels, cmap='jet')
        # plt.pcolormesh(params.x, params.z, v_slice_ycheck.T, cmap='bwr')
        # plt.colorbar()
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$z$")
        # plt.xlim(np.min(params.x), np.max(params.x))
        # plt.ylim(np.min(params.z), np.max(params.z))
        # plt.title(f"Normal velocity v at t={t_check}, y={params.y_check}")
        # plt.tight_layout()
        # plt.show()
        
        # plt.figure(figsize=(10, 6))
        # # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
        # cs = plt.contour(params.x, params.z, v_slice_ycheck.T, colors='black', linewidths=0.6)
        # # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
        # # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
        # # plt.colorbar()
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$z$")
        # plt.xlim(np.min(params.x), np.max(params.x))
        # plt.ylim(np.min(params.z), np.max(params.z))
        # plt.title(f"Contours of normal velocity v at t={t_check}, y={params.y_check}")
        # plt.tight_layout()
        # plt.show()
        
        # plt.figure(figsize=(10,6))
        # # plt.contourf(x, z, v_slice_ycheck.T, levels=levels, cmap='jet')
        # plt.pcolormesh(params.x, params.z, v_fitted_slice_ycheck.T, cmap='bwr')
        # plt.colorbar()
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$z$")
        # plt.xlim(np.min(params.x), np.max(params.x))
        # plt.ylim(np.min(params.z), np.max(params.z))
        # plt.title(f"Fitted normal velocity v at t={t_check}, y={params.y_check}")
        # plt.tight_layout()
        # plt.show()
        
        # plt.figure(figsize=(10, 6))
        # # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
        # cs = plt.contour(params.x, params.z, v_fitted_slice_ycheck.T, colors='black', linewidths=0.6)
        # # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
        # # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
        # # plt.colorbar()
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$z$")
        # plt.xlim(np.min(params.x), np.max(params.x))
        # plt.ylim(np.min(params.z), np.max(params.z))
        # plt.title(f"Contours of fitted normal velocity v at t={t_check}, y={params.y_check}")
        # plt.tight_layout()
        # plt.show()
        
    plt.figure(figsize=(10,6))
    for i in range (len(params.t_check_list_POD)):
        plt.plot(params.x, eta_fitted_slice_ycheck_all_z_centered[:, i], label=f"t={params.t_check_list_POD[i]}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"Fitted normal vorticity at y={}".format(params.y_check))
    plt.title("Fitted normal vorticity at y={} over time".format(params.y_check))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,6))
    for i in range (len(params.t_check_list_POD)):
        plt.plot(params.x, eta_slice_ycheck_all_z_centered[:, i], label=f"t={params.t_check_list_POD[i]}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"Normal vorticity at y={}".format(params.y_check))
    plt.title("Normal vorticity at y={} over time".format(params.y_check))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,6))
    for i in range (len(params.t_check_list_POD)):
        plt.plot(params.x, deriv_eta_fitted_slice_ycheck_all_z_centered[:, i], label=f"t={params.t_check_list_POD[i]}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"Fitted RHS of the normal vorticity at y={}".format(params.y_check))
    plt.title("Fitted RHS of the normal vorticity at y={} over time".format(params.y_check))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,6))
    for i in range (len(params.t_check_list_POD)):
        plt.plot(params.x, deriv_eta_slice_ycheck_all_z_centered[:, i], label=f"t={params.t_check_list_POD[i]}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"RHS of the normal vorticity at y={}".format(params.y_check))
    plt.title("RHS of the normal vorticity at y={} over time".format(params.y_check))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# endregion





