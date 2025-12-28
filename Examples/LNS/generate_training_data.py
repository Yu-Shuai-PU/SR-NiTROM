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
2. Successfully verify that the disturbance kinetic energy computed using (u, v, w) and (v, eta) are consistent.
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
fname_traj_template_dx = data_path + "traj_template_dx.npy"
fname_traj_template_dxx = data_path + "traj_template_dxx.npy"
fname_traj_init = data_path + "traj_init_%03d.npy" # for initial condition of u
fname_traj_init_fitted = data_path + "traj_init_fitted_%03d.npy" # for initial condition of u fitted
fname_traj = traj_path + "traj_%03d.npy" # for u
fname_traj_fitted = traj_path + "traj_fitted_%03d.npy" # for u fitted
fname_weight_traj = traj_path + "weight_traj_%03d.npy"
fname_weight_shift_amount = traj_path + "weight_shift_amount_%03d.npy"
fname_weight_shift_speed = traj_path + "weight_shift_speed_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy" # for du/dt
fname_deriv_fitted = traj_path + "deriv_fitted_%03d.npy" # for du/dt fitted
fname_shift_amount = traj_path + "shift_amount_%03d.npy" # for shifting amount
fname_shift_speed = traj_path + "shift_speed_%03d.npy" # for shifting speed
fname_time = traj_path + "time.npy"

### First, we try to set up the initial localized disturbance

Lx = 48
Ly = 2 # from -1 to 1
Lz = 24

nx = 96
ny = 65 # ny includes the boundary points when using Chebyshev grid
nz = 96

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

traj_template = np.load(fname_traj_template)
traj_template_dx = np.load(fname_traj_template_dx)

fom = fom_class_LNS.LNS(Lx, Ly, Lz, nx, ny, nz, y, Re, U_base, U_base_dy, U_base_dyy)
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
psi0[:, :, :, 0] = amp * (1-Y**2)**2 * ((X - Lx/2)/2) * (Z - Lz/2) ** 2 * np.exp(-((X - Lx/2)/2)**2 - ((Z - Lz/2)/2)**2)

for k in range (n_traj):
    v0[:, :, :, k] = fom.diff_z(psi0[:, :, :, k], order = 1)
    eta0[:, :, :, k] = fom.diff_x(fom.diff_1_y(psi0[:, :, :, k]), order = 1)

# endregion

# region 2: Simulations

pool_inputs = (MPI.COMM_WORLD, n_traj)
pool = classes.mpi_pool(*pool_inputs)

# Before doing any data generation, we first run on the benchmark training trajectory to get the trajectory template

# for k in range (pool.my_n_traj):
#     traj_idx = k + pool.disps[pool.rank]
#     print("Running simulation %d/%d"%(traj_idx,n_traj - 1))
#     traj_init = np.concatenate((v0[:, :, :, traj_idx].ravel(), eta0[:, :, :, traj_idx].ravel()))
    
#     traj, tsave = tstep_kse_fom.time_step(traj_init, nsave) # traj is in the physical domain and is of the shape (2 * nx * ny * nz, nsave_samples)
#     traj_fitted, shifting_amount = fom.template_fitting(traj)
#     traj_init_fitted = traj_fitted[:, 0]
#     deriv = np.zeros_like(traj)
#     deriv_fitted = np.zeros_like(traj_fitted)
#     shifting_speed = np.zeros(len(tsave))
#     disturbance_energy = np.zeros(len(tsave))
#     for i in range (traj.shape[1]):
#         deriv[:, i] = fom.evaluate_fom_rhs_unreduced(traj[:, i])
#         deriv_v_3D = deriv[:nx * ny * nz, i].reshape((nx, ny, nz))
#         deriv_eta_3D = deriv[nx * ny * nz:, i].reshape((nx, ny, nz))
#         deriv_v_3D_fitted = fom.shift_x_input_3D(deriv_v_3D, -shifting_amount[i])
#         deriv_eta_3D_fitted = fom.shift_x_input_3D(deriv_eta_3D, -shifting_amount[i])
#         deriv_fitted[:, i] = np.concatenate((deriv_v_3D_fitted.ravel(), deriv_eta_3D_fitted.ravel()))
#         shifting_speed[i] = fom.evaluate_fom_shifting_speed(traj_fitted[:, i], deriv_fitted[:, i])
#         disturbance_energy[i] = fom.inner_product_3D(traj[:, i][0 : nx * ny * nz].reshape((nx, ny, nz)),
#                                                      traj[:, i][nx * ny * nz : ].reshape((nx, ny, nz)),
#                                                      traj[:, i][0 : nx * ny * nz].reshape((nx, ny, nz)),
#                                                      traj[:, i][nx * ny * nz : ].reshape((nx, ny, nz)))
        
#     weight_traj = 1.0/np.mean(disturbance_energy)
#     weight_shifting_amount = 1.0/Lx**2
#     weight_shifting_speed = 1.0/np.mean((shifting_speed - np.mean(shifting_speed))**2)
        
#     plt.plot(tsave, shifting_amount)
#     plt.xlabel("Time")
#     plt.ylabel("Shifting amount c(t)")
#     plt.title("Shifting amount over time")
#     plt.tight_layout()
#     plt.show()
    
#     plt.plot(tsave, shifting_speed)
#     plt.xlabel("Time")
#     plt.ylabel("Shifting speed c'(t)")
#     plt.title("Shifting speed over time")
#     plt.tight_layout()
#     plt.show()
        
#     np.save(fname_time, tsave)
#     np.save(fname_traj_init%traj_idx,traj_init)
#     np.save(fname_traj_init_fitted%traj_idx,traj_init_fitted)
#     np.save(fname_traj%traj_idx, traj)
#     np.save(fname_traj_fitted%traj_idx, traj_fitted)
#     np.save(fname_deriv%traj_idx, deriv)
#     np.save(fname_deriv_fitted%traj_idx, deriv_fitted)
#     np.save(fname_shift_amount%traj_idx, shifting_amount)
#     np.save(fname_shift_speed%traj_idx, shifting_speed)
#     np.save(fname_weight_traj%traj_idx, weight_traj)
#     np.save(fname_weight_shift_amount%traj_idx, weight_shifting_amount)
#     np.save(fname_weight_shift_speed%traj_idx, weight_shifting_speed)
    
#     traj_v = traj[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
#     traj_eta = traj[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
#     traj_v_fitted = traj_fitted[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
#     traj_eta_fitted = traj_fitted[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
#     deriv_v = deriv[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
#     deriv_eta = deriv[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
#     deriv_v_fitted = deriv_fitted[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
#     deriv_eta_fitted = deriv_fitted[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
    
#     t_check_list = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]
#     y_check = -0.56
    
#     v_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
#     eta_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
#     v_fitted_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
#     eta_fitted_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
#     deriv_v_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
#     deriv_eta_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
#     deriv_v_fitted_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))
#     deriv_eta_fitted_slice_ycheck_all_z_centered = np.zeros((nx, len(t_check_list)))

#     for t_check in t_check_list:

#         idx_sample = int(t_check / (dt * nsave))
#         v_slice = traj_v[:, :, :, idx_sample]
#         eta_slice = traj_eta[:, :, :, idx_sample]
#         v_fitted_slice = traj_v_fitted[:, :, :, idx_sample]
#         eta_fitted_slice = traj_eta_fitted[:, :, :, idx_sample]
#         deriv_v_slice = deriv_v[:, :, :, idx_sample]
#         deriv_eta_slice = deriv_eta[:, :, :, idx_sample]
#         deriv_v_fitted_slice = deriv_v_fitted[:, :, :, idx_sample]
#         deriv_eta_fitted_slice = deriv_eta_fitted[:, :, :, idx_sample]
#         idx_y_check = np.argmin(np.abs(y - y_check))
#         v_slice_ycheck = v_slice[:, idx_y_check, :]
#         v_fitted_slice_ycheck = v_fitted_slice[:, idx_y_check, :]
#         eta_slice_ycheck = eta_slice[:, idx_y_check, :]
#         eta_fitted_slice_ycheck = eta_fitted_slice[:, idx_y_check, :]
#         deriv_v_slice_ycheck = deriv_v_slice[:, idx_y_check, :]
#         deriv_v_fitted_slice_ycheck = deriv_v_fitted_slice[:, idx_y_check, :]
#         deriv_eta_slice_ycheck = deriv_eta_slice[:, idx_y_check, :]
#         deriv_eta_fitted_slice_ycheck = deriv_eta_fitted_slice[:, idx_y_check, :]
        
#         v_fitted_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = v_fitted_slice_ycheck[:, nz//2]
#         eta_fitted_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = eta_fitted_slice_ycheck[:, nz//2]
#         v_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = v_slice_ycheck[:, nz//2]
#         eta_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = eta_slice_ycheck[:, nz//2]
#         deriv_v_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = deriv_v_slice_ycheck[:, nz//2]
#         deriv_eta_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = deriv_eta_slice_ycheck[:, nz//2]
#         deriv_v_fitted_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = deriv_v_fitted_slice_ycheck[:, nz//2]
#         deriv_eta_fitted_slice_ycheck_all_z_centered[:, t_check_list.index(t_check)] = deriv_eta_fitted_slice_ycheck[:, nz//2]  

#         v_min = np.min(v_slice_ycheck)
#         v_max = np.max(v_slice_ycheck)
#         # v_spacing = 1e-6  # 等高线间距
        
#         eta_min = np.min(eta_slice_ycheck)
#         eta_max = np.max(eta_slice_ycheck)
#         # eta_spacing = 1e-6  # 等高线间距

#         # 构造等高线 levels
#         # levels = np.arange(v_min - v_spacing, v_max + v_spacing, v_spacing)
#         # plt.figure(figsize=(10,6))
#         # # plt.contourf(x, z, v_slice_ycheck.T, levels=levels, cmap='jet')
#         # plt.pcolormesh(x, z, v_slice_ycheck.T, cmap='bwr')
#         # plt.colorbar()
#         # plt.xlabel(r"$x$")
#         # plt.ylabel(r"$z$")
#         # plt.xlim(np.min(x), np.max(x))
#         # plt.ylim(np.min(z), np.max(z))
#         # plt.title(f"Normal velocity v at t={t_check}, y={y_check}")
#         # plt.tight_layout()
#         # plt.show()
        
#         # plt.figure(figsize=(10, 6))
#         # # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
#         # cs = plt.contour(x, z, v_slice_ycheck.T, colors='black', linewidths=0.6)
#         # # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
#         # # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
#         # # plt.colorbar()
#         # plt.xlabel(r"$x$")
#         # plt.ylabel(r"$z$")
#         # plt.xlim(np.min(x), np.max(x))
#         # plt.ylim(np.min(z), np.max(z))
#         # plt.title(f"Contours of normal velocity v at t={t_check}, y={y_check}")
#         # plt.tight_layout()
#         # plt.show()
        
#         # plt.figure(figsize=(10,6))
#         # # plt.contourf(x, z, v_slice_ycheck.T, levels=levels, cmap='jet')
#         # plt.pcolormesh(x, z, v_fitted_slice_ycheck.T, cmap='bwr')
#         # plt.colorbar()
#         # plt.xlabel(r"$x$")
#         # plt.ylabel(r"$z$")
#         # plt.xlim(np.min(x), np.max(x))
#         # plt.ylim(np.min(z), np.max(z))
#         # plt.title(f"Fitted normal velocity v at t={t_check}, y={y_check}")
#         # plt.tight_layout()
#         # plt.show()
        
#         # plt.figure(figsize=(10, 6))
#         # # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
#         # cs = plt.contour(x, z, v_fitted_slice_ycheck.T, colors='black', linewidths=0.6)
#         # # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
#         # # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
#         # # plt.colorbar()
#         # plt.xlabel(r"$x$")
#         # plt.ylabel(r"$z$")
#         # plt.xlim(np.min(x), np.max(x))
#         # plt.ylim(np.min(z), np.max(z))
#         # plt.title(f"Contours of fitted normal velocity v at t={t_check}, y={y_check}")
#         # plt.tight_layout()
#         # plt.show()
        
#     plt.figure(figsize=(10,6))
#     for i in range (len(t_check_list)):
#         plt.plot(x, eta_fitted_slice_ycheck_all_z_centered[:, i], label=f"t={t_check_list[i]}")
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"Fitted normal vorticity at y={}".format(y_check))
#     plt.title("Fitted normal vorticity at y={} over time".format(y_check))
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    
#     plt.figure(figsize=(10,6))
#     for i in range (len(t_check_list)):
#         plt.plot(x, eta_slice_ycheck_all_z_centered[:, i], label=f"t={t_check_list[i]}")
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"Normal vorticity at y={}".format(y_check))
#     plt.title("Normal vorticity at y={} over time".format(y_check))
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    
#     plt.figure(figsize=(10,6))
#     for i in range (len(t_check_list)):
#         plt.plot(x, deriv_eta_slice_ycheck_all_z_centered[:, i], label=f"t={t_check_list[i]}")
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"RHS of the normal vorticity at y={}".format(y_check))
#     plt.title("RHS of the normal vorticity at y={} over time".format(y_check))
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    
#     plt.figure(figsize=(10,6))
#     for i in range (len(t_check_list)):
#         plt.plot(x, deriv_eta_fitted_slice_ycheck_all_z_centered[:, i], label=f"t={t_check_list[i]}")
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"Fitted RHS of the normal vorticity at y={}".format(y_check))
#     plt.title("Fitted RHS of the normal vorticity at y={} over time".format(y_check))
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    
### Now testing: we try to load these data and perform POD on the template-fitted trajectory data
    
#%% # Generate and save trajectory
pool_inputs = (MPI.COMM_WORLD, n_traj)
pool_kwargs = {'fname_time':fname_time, 'fname_traj':fname_traj,'fname_traj_fitted':fname_traj_fitted,
               'fname_X_template':fname_traj_template, 'fname_X_template_dx':fname_traj_template_dx, 'fname_X_template_dxx':fname_traj_template_dxx,
               'fname_deriv':fname_deriv,'fname_deriv_fitted':fname_deriv_fitted,
               'fname_shift_amount':fname_shift_amount,'fname_shift_speed':fname_shift_speed}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)
pool.load_data()

T_final = pool.time[-1]

r = 20 # ROM dimension, should account for 99.5% energy
timespan_percentage_POD = 1 # percentage of the entire timespan used for POD
snapshot_start_time_idx_POD = 0
snapshot_end_time_idx_POD = 1 + int(timespan_percentage_POD * (pool.n_snapshots - 1))
max_iterations = 20
leggauss_deg = 5
nsave_rom = 11 # nsave_rom = 1 + int(dt_sample/dt) = 1 + sample_interval
which_trajs = np.arange(0, pool.my_n_traj, 1)
which_times = np.arange(snapshot_start_time_idx_POD,snapshot_end_time_idx_POD,1)

poly_comp = [1] # polynomial degree for the ROM dynamics

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

# def verify_correlation_matrix(fom, X_full, num_test=5):
#     """
#     对拍验证函数：对比矩阵化计算与逐个积分计算的结果
    
#     参数:
#     fom: 你的 FOM 对象 (包含 compute_snapshot_correlation_matrix 和 inner_product_3D)
#     X_full: 全量快照矩阵 (2*nx*ny*nz, Total_Snapshots)
#     num_test: 测试的快照数量 (建议 5-10 个，多了跑得慢)
#     """
    
#     print(f"🔍 开始验证 (测试快照数: {num_test})...")
    
#     # 1. 随机挑选几个快照 (或者直接取前 N 个)
#     # 确保 X_test 是 (2*grid, num_test) 形状
#     indices = np.random.choice(X_full.shape[1], num_test, replace=False)
#     # indices = np.arange(num_test) # 或者直接用前几个
#     X_test = X_full[:, indices]
    
#     # ---------------------------------------------------------
#     # 方法 A: 矩阵化极速计算 (The "Black Box")
#     # ---------------------------------------------------------
#     t0 = timer.time()
#     C_fast = fom.compute_snapshot_correlation_matrix(X_test)
#     t_fast = timer.time() - t0
#     print(f"⚡ 矩阵化方法耗时: {t_fast:.6f} s")
    
#     # ---------------------------------------------------------
#     # 方法 B: 双重循环逐个计算 (The "Ground Truth")
#     # ---------------------------------------------------------
#     t0 = timer.time()
#     C_slow = np.zeros((num_test, num_test))
    
#     # 准备解包数据，确保传给 inner_product_3D 的数据格式一模一样
#     num_grid = fom.nx * fom.ny * fom.nz
#     nx, ny, nz = fom.nx, fom.ny, fom.nz
    
#     # 预先 reshape 所有快照，模拟 inner_product_3D 需要的输入
#     snapshots_v = []
#     snapshots_eta = []
    
#     for k in range(num_test):
#         # 注意这里 reshape 的顺序要和 fom 里完全一致
#         # fom里是: q_vec[:num_grid, :].T.reshape(M, nx, ny, nz)
#         # 所以这里单向量 reshape 应该是:
#         q_vec = X_test[:, k]
#         v_field = q_vec[:num_grid].reshape(nx, ny, nz)
#         eta_field = q_vec[num_grid:].reshape(nx, ny, nz)
        
#         snapshots_v.append(v_field)
#         snapshots_eta.append(eta_field)
        
#     print(f"🐢 逐个积分类似中 (共 {num_test*num_test} 次内积)...")
    
#     for i in range(num_test):
#         for j in range(num_test):
#             # 计算 <q_i, q_j>_W
#             # 注意 inner_product_3D 的参数顺序，通常是 (v1, eta1, v2, eta2)
#             val = fom.inner_product_3D(snapshots_v[i], snapshots_eta[i], 
#                                        snapshots_v[j], snapshots_eta[j])
#             C_slow[i, j] = val
            
#     t_slow = timer.time() - t0
#     print(f"🐢 逐个积分耗时: {t_slow:.6f} s (加速比: {t_slow/t_fast:.1f}x)")
    
#     # ---------------------------------------------------------
#     # 结果对比
#     # ---------------------------------------------------------
#     diff = np.abs(C_fast - C_slow)
#     max_diff = np.max(diff)
#     rel_diff = max_diff / (np.max(np.abs(C_slow)) + 1e-16)
    
#     print("-" * 40)
#     print(f"最大绝对误差 (Max Abs Diff): {max_diff:.4e}")
#     print(f"最大相对误差 (Max Rel Diff): {rel_diff:.4e}")
    
#     if np.allclose(C_fast, C_slow, rtol=1e-10, atol=1e-12):
#         print("✅ 验证通过！两个函数结果完全一致。")
#         print("   证明：compute_snapshot_correlation_matrix 正确处理了非对角权重。")
#     else:
#         print("❌ 验证失败！结果不一致。")
#         print("   请检查 reshape 顺序 (Fortran vs C order) 或 导数计算逻辑。")
#         print("   对比矩阵切片 (前2x2):")
#         print("   Fast:\n", C_fast[:2,:2])
#         print("   Slow:\n", C_slow[:2,:2])

# # ==========================================
# # 运行验证
# # ==========================================
# # 假设 opt_obj.X_fitted 是你的数据，fom 是你的对象
# # 只需要取少量数据验证即可 (比如 Rank 0 的前 10 个快照)
# # 注意：X_fitted 可能是 (Rank, Space, Time)，这里我们需要 (Space, Time)
# # 如果你是单处理器，可以直接用:
# data_for_test = np.ascontiguousarray(opt_obj.X_fitted[0, :, :]) 

# verify_correlation_matrix(fom, data_for_test, num_test=5)

Phi_POD, cumulative_energy_proportion = opinf_fun.perform_POD(pool,opt_obj,r,fom)
Psi_POD = Phi_POD.copy()

## Test the weighted orthogonality of Phi_POD:

# for i in range (r):
#     for j in range (r):
#         ip_ij = fom.inner_product_3D(Phi_POD[0 : nx * ny * nz, i].reshape((nx, ny, nz)),
#                                      Phi_POD[nx * ny * nz : , i].reshape((nx, ny, nz)),
#                                      Phi_POD[0 : nx * ny * nz, j].reshape((nx, ny, nz)),
#                                      Phi_POD[nx * ny * nz : , j].reshape((nx, ny, nz)))
#         if i == j:
#             print(f"Weighted inner product of POD mode {i} with itself: {ip_ij:.4e} (should be 1.0)")
#         else:
#             print(f"Weighted inner product of POD mode {i} with POD mode {j}: {ip_ij:.4e} (should be 0.0)")

# X_data = np.ascontiguousarray(opt_obj.X_fitted[0, :, :]) 
# N_snapshots = X_data.shape[1]
# nx, ny, nz = fom.nx, fom.ny, fom.nz
# num_grid = nx * ny * nz

# print(f"开始全局验证 (共 {N_snapshots} 个快照)...")
# print("注意：因为使用了显式循环调用 inner_product_3D，可能会花一点时间，请耐心等待。")

# # 1. 预处理 POD 模态 (切片并 Reshape，避免循环里重复做)
# # ------------------------------------------------------
# phis_v = []
# phis_eta = []
# for k in range(r):
#     phi_k = Phi_POD[:, k]
#     phis_v.append(phi_k[:num_grid].reshape(nx, ny, nz))
#     phis_eta.append(phi_k[num_grid:].reshape(nx, ny, nz))

# # 2. 初始化能量累加器
# # ------------------------------------------------------
# total_energy_sq_sum = 0.0  # sum( ||q||^2 )
# error_energy_sq_sum = 0.0  # sum( ||q - q_rec||^2 )

# t_start = timer.time()

# # 3. 时间循环
# # ------------------------------------------------------
# for t in range(N_snapshots):
#     # A. 提取当前快照
#     q_vec = X_data[:, t]
#     q_v = q_vec[:num_grid].reshape(nx, ny, nz)
#     q_eta = q_vec[num_grid:].reshape(nx, ny, nz)
    
#     # B. 计算原始能量 ||q||_W^2
#     norm_q_sq = fom.inner_product_3D(q_v, q_eta, q_v, q_eta)
#     total_energy_sq_sum += norm_q_sq
    
#     # C. 正确投影 (Project) -> 得到系数 a_k
#     # a_k = <q, phi_k>_W
#     coeffs = np.zeros(r)
#     for k in range(r):
#         coeffs[k] = fom.inner_product_3D(q_v, q_eta, phis_v[k], phis_eta[k])
    
#     # D. 正确重构 (Reconstruct) -> 得到 q_rec
#     # q_rec = sum(a_k * phi_k)
#     # 利用矩阵乘法加速线性组合: (2N, r) @ (r,) -> (2N,)
#     q_rec_vec = Phi_POD @ coeffs
    
#     # E. 计算误差能量 ||q - q_rec||_W^2
#     diff_vec = q_vec - q_rec_vec
#     diff_v = diff_vec[:num_grid].reshape(nx, ny, nz)
#     diff_eta = diff_vec[num_grid:].reshape(nx, ny, nz)
    
#     norm_err_sq = fom.inner_product_3D(diff_v, diff_eta, diff_v, diff_eta)
#     error_energy_sq_sum += norm_err_sq
    
#     # 打印进度
#     if (t+1) % 10 == 0 or (t+1) == N_snapshots:
#         print(f"  Processing snapshot {t+1}/{N_snapshots}...", end='\r')

# print(f"\n计算完成。耗时: {timer.time() - t_start:.2f} s")

# # 4. 最终计算与对比
# # ------------------------------------------------------
# # 实际相对误差
# global_rel_error = np.sqrt(error_energy_sq_sum) / np.sqrt(total_energy_sq_sum)

# # 理论相对误差 (基于特征值)
# # cumulative_energy_proportion 是百分比 (比如 99.99)，要除以 100
# energy_captured_ratio = cumulative_energy_proportion[-1] / 100.0
# theoretical_error = np.sqrt(1.0 - energy_captured_ratio)

# print("\n" + "="*50)
# print("🏆 全局物理误差与理论值终极对决")
# print("="*50)
# print(f"POD 能量捕捉率: {energy_captured_ratio*100:.4f}%")
# print("-" * 50)
# print(f"理论相对误差 (sqrt(1-Energy)): {theoretical_error:.6e}")
# print(f"实际全场误差 (Global Norm):    {global_rel_error:.6e}")
# print("-" * 50)

# diff = abs(global_rel_error - theoretical_error)
# if diff < 1e-4: # 稍微放宽一点点，因为特征值求和与积分求和有极其微小的数值精度差异
#     print("✅ 完美吻合！(差异 < 1e-4)")
#     print("   证明：Phi_POD 是完美的，投影和重构逻辑在物理上完全闭环。")
# else:
#     print("⚠️ 仍有差异。")
#     print("   如果差异很小 (e.g. 1e-3)，可能是积分精度问题。")
#     print("   如果差异很大，那天塌了。")
# print("="*50)

# Phi_test = np.random.randn(2 * fom.nx * fom.ny * fom.nz, 2)

# # 1. 用 compute_W_times_basis 计算矩阵形式结果
# W_Phi = fom.compute_W_times_basis(Phi_test)
# Matrix_Result = Phi_test.T @ W_Phi  # 应该是 2x2 矩阵

# # 2. 用 inner_product_3D 计算真值
# True_Result = np.zeros((2, 2))
# # 需要手动拆解 Phi_test 为 v, eta 传给 inner_product
# grid = fom.nx * fom.ny * fom.nz
# for i in range(2):
#     for j in range(2):
#         q1 = Phi_test[:, i]
#         q2 = Phi_test[:, j]
#         # 拆解
#         v1 = q1[:grid].reshape(fom.nx, fom.ny, fom.nz)
#         eta1 = q1[grid:].reshape(fom.nx, fom.ny, fom.nz)
#         v2 = q2[:grid].reshape(fom.nx, fom.ny, fom.nz)
#         eta2 = q2[grid:].reshape(fom.nx, fom.ny, fom.nz)
        
#         True_Result[i, j] = fom.inner_product_3D(v1, eta1, v2, eta2)

# # 3. 对比
# print("Difference:", np.linalg.norm(Matrix_Result - True_Result))
# if np.allclose(Matrix_Result, True_Result):
#     print("✅ 完美！compute_W_times_basis 正确实现了 inner_product_3D 的逻辑！")
# else:
#     print("❌ 还有 Bug...")

# W_Phi_POD = fom.compute_W_times_basis(Phi_POD)
# orthogonality_matrix = Phi_POD.T @ W_Phi_POD
# print(f"relative diff from weighted orthogonality: {np.linalg.norm(orthogonality_matrix - np.eye(r)) / np.linalg.norm(np.eye(r)):.4e}")
# print(f"relative diff from orthogonality (Euclidean): {np.linalg.norm(Phi_POD.T @ Phi_POD - np.eye(r)) / np.linalg.norm(np.eye(r)):.4e}")

# plt.semilogy(np.arange(1, len(cumulative_energy_proportion) + 1), cumulative_energy_proportion, marker='o')
# plt.xlabel("Number of POD modes")
# plt.ylabel("Cumulative Energy Proportion")
# plt.title("Cumulative Energy Proportion vs Number of POD Modes (Snapshot Method)")
# plt.grid()
# plt.tight_layout()
# plt.show()

Psi_POD, PhiF_POD = fom.generate_weighted_projection_bases(Phi_POD, Psi_POD)

print(f"relative difference between PhiF_POD and Phi_POD: {np.linalg.norm(PhiF_POD - Phi_POD) / np.linalg.norm(Phi_POD):.4e}")
### Test the reconstruction accuracy of POD basis

traj_fitted = opt_obj.X_fitted[0,:,:]

traj_fitted_proj = Psi_POD.T @ traj_fitted
traj_fitted_recon = PhiF_POD @ traj_fitted_proj

# 1. 计算原始误差场 (Difference)
diff_matrix = traj_fitted - traj_fitted_recon

# 2. 计算 W * Diff 和 W * Original
# 利用现有的函数计算加权后的矩阵
# compute_W_times_basis 既然能算 Basis (2N, r)，当然也能算 Snapshots (2N, M)
W_diff_matrix = fom.compute_W_times_basis(diff_matrix)
W_traj_fitted = fom.compute_W_times_basis(traj_fitted)

# 3. 计算加权范数 (Weighted Norms)
# 技巧：||A||_W^2 = sum(A * (W*A))。
# 这里的 sum 是对矩阵所有元素求和 (相当于 trace(A.T @ W @ A))

# 误差的加权能量
norm_error_W = np.sqrt(np.sum(diff_matrix * W_diff_matrix))

# 原始流场的加权能量
norm_traj_W  = np.sqrt(np.sum(traj_fitted * W_traj_fitted))

# 4. 计算相对误差
rel_error_W = norm_error_W / norm_traj_W

print(f"Standard (Euclidean) Error: {np.linalg.norm(diff_matrix) / np.linalg.norm(traj_fitted):.4e}")
print(f"Weighted (Physical) Error:  {rel_error_W:.4e}")

...

### Test the SR-Galerkin ROM simulation accuracy

# Tensors_POD = fom.assemble_petrov_galerkin_tensors(PhiF_POD, Psi_POD) # A, p, s, M

# for k in range(pool.my_n_traj):
#     traj_idx = k + pool.disps[pool.rank]
#     print("Preparing SR-Galerkin simulation %d/%d"%(traj_idx,pool.n_traj))
#     traj_SRG_init = Psi_POD.T@opt_obj.X_fitted[k,:,0].reshape(-1)
#     shifting_amount_SRG_init = opt_obj.c[k,0]

#     sol_SRG = solve_ivp(opt_obj.evaluate_rom_rhs,
#                     [opt_obj.time[0],opt_obj.time[-1]],
#                     np.hstack((traj_SRG_init, shifting_amount_SRG_init)),
#                     'RK45',
#                     t_eval=opt_obj.time,
#                     args=(np.zeros(r),) + Tensors_POD).y
    
#     traj_fitted_SRG = PhiF_POD@sol_SRG[:-1,:]
#     traj_fitted_SRG_v = traj_fitted_SRG[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
#     traj_fitted_SRG_eta = traj_fitted_SRG[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
#     shifting_amount_SRG = sol_SRG[-1,:]
#     traj_SRG = np.zeros_like(traj_fitted_SRG)
#     shifting_speed_SRG = np.zeros_like(shifting_amount_SRG)

#     for j in range (len(opt_obj.time)):
#         traj_SRG_v_vec = fom.shift_x_input_3D(traj_fitted_SRG_v[:, :, :, j], shifting_amount_SRG[j])
#         traj_SRG_eta_vec = fom.shift_x_input_3D(traj_fitted_SRG_eta[:, :, :, j], shifting_amount_SRG[j])
#         traj_SRG[:,j] = np.concatenate((traj_SRG_v_vec.ravel(), traj_SRG_eta_vec.ravel()))
#         shifting_speed_SRG[j] = opt_obj.compute_shift_speed(sol_SRG[:-1,j], Tensors_POD)
#         relative_error[j] = ...
        
#     relative_error_space_time_SRG[traj_idx] = np.linalg.norm(opt_obj.X[k,:,:] - X_SRG)/np.linalg.norm(opt_obj.X[k,:,:])
#     X_FOM = opt_obj.X[k,:,:]
#     c_FOM = opt_obj.c[k,:]
#     cdot_FOM = opt_obj.cdot[k,:]
#     X_fitted_FOM = opt_obj.X_fitted[k,:,:]
        
#     np.save(fname_traj_FOM%traj_idx,X_FOM)
#     np.save(fname_traj_fitted_FOM%traj_idx,X_fitted_FOM)
#     np.save(fname_shift_amount_FOM%traj_idx,c_FOM)
#     np.save(fname_shift_speed_FOM%traj_idx,cdot_FOM)    
#     np.save(fname_traj_SRG%traj_idx,X_SRG)
#     np.save(fname_traj_fitted_SRG%traj_idx,X_fitted_SRG)
#     np.save(fname_shift_amount_SRG%traj_idx,c_SRG)
#     np.save(fname_shift_speed_SRG%traj_idx,cdot_SRG)
#     np.save(fname_relative_error_SRG%traj_idx,relative_error)
    
#     ### Plotting, things to be done:
#     ### 1. switch from contourf to pcolormesh
#     ### 2. apply Fourier spectral interpolation to plot a 40-mode solution on a 256-point grid for better visualization
    
#     plt.figure(figsize=(10,6))
#     plt.contourf(x,opt_obj.time,opt_obj.X[k,:,:].T, levels = np.linspace(-16, 16, 9), cmap=cmap_name)
#     plt.colorbar()
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$t$")
#     plt.tight_layout()
#     if traj_idx == 0:
#         plt.title(f"FOM solution, initial condition = uIC")
#     else:
#         plt.title(f"FOM solution, initial condition = uIC + {amp_array[traj_idx - 1]} * {training_ic_perturbation_name}")
#     plt.savefig(fig_path + "traj_FOM_%03d.png"%traj_idx)
#     plt.close()

#     plt.figure(figsize=(10,6))
#     plt.contourf(x,opt_obj.time,X_SRG.T, levels = np.linspace(-16, 16, 9), cmap=cmap_name)
#     plt.colorbar()
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$t$")
#     plt.tight_layout()
#     if traj_idx == 0:
#         plt.title(f"SRG solution, error: {relative_error_space_time_SRG[k]:.4e}, initial condition = uIC")
#     else:
#         plt.title(f"SRG solution, error: {relative_error_space_time_SRG[k]:.4e}, initial condition = uIC + {amp_array[traj_idx - 1]} * {training_ic_perturbation_name}")
#     plt.savefig(fig_path + "traj_SRG_%03d.png"%traj_idx)
#     plt.close()
    
# relative_error_space_time_SRG = np.sum(np.asarray(pool.comm.allgather(relative_error_space_time_SRG)), axis=0)


# endregion





