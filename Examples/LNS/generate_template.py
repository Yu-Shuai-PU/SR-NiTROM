import numpy as np 
import scipy 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpi4py import MPI

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys
import os
# plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
# plt.rc('text.latex',preamble=r'\usepackage{amsmath}')
sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

import configs
import fom_class_LNS
import func_plot

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
params = configs.load_configs()
fom = fom_class_LNS.LNS(params.Lx, params.Ly, params.Lz, 
                        params.nx, params.ny, params.nz,
                        params.y, params.Re,
                        params.U_base, params.U_base_dy, params.U_base_dyy)
tstep_kse_fom = fom_class_LNS.time_step_LNS(fom, params.time)

v0 = fom.diff_z(params.psi0[:, :, :, 0], order = 1)
eta0 = fom.diff_x(fom.diff_1_y(params.psi0[:, :, :, 0]), order = 1)

# endregion

# region 2: Simulations

# Before doing any data generation, we first run on the benchmark training trajectory to get the trajectory template

print("Running the benchmark simulation")
traj_init = np.concatenate((v0.ravel(), eta0.ravel()))
traj, tsave = tstep_kse_fom.time_step(traj_init, params.nsave) # traj is in the physical domain and is of the shape (2 * nx * ny * nz, nsave_samples)
traj_PSD = fom.compute_PSD(traj) ### Compute the Power Spectral Density (PSD) of the trajectory for various wavenumbers (kx, kz), which will be a pcolormesh plot later of (kx, kz, PSD(kx, kz, N_snapshots))

kx = fom.kx
kz = fom.kz
KX, KZ = np.meshgrid(kx, kz, indexing='ij')
fig, ax = plt.subplots(figsize=(10, 10))

# --- 1. 定义截断框 (保持不变, ratio = 2 是不进行截断) ---
ratio = 2
kxc = (params.nx // ratio) * (2*np.pi/fom.Lx)
kzc = (params.nz // ratio) * (2*np.pi/fom.Lz)
# 将这些坐标打包，准备传给 update 函数
box_coords = ([-kxc, kxc, kxc, -kxc, -kxc], [-kzc, -kzc, kzc, kzc, -kzc])

levels = [1e-16, 1e-14, 1e-12, 1e-10, 1e-6, 1e-4, 1e-2, 0.1, 0.5]

# --- 2. 修改 update 函数以接收参数 ---
def update(frame, rect_x, rect_z): # 添加了参数接收
    ax.clear()
    
    psd = traj_PSD[:, :, frame]
    # 归一化：除以当前时刻总能量，关注“形状”
    total_E = np.sum(psd)
    if total_E > 0:
        psd_norm = psd / total_E
    else:
        psd_norm = psd
        
    # 绘图
    CS = ax.contour(KX, KZ, psd_norm, levels=levels, colors='k', linewidths=0.5)
    ax.clabel(CS, inline=1, fontsize=8, fmt='%1.0e')
    ax.contourf(KX, KZ, psd_norm, levels=levels, cmap='inferno_r', alpha=0.5)
    
    # 画出截断框 (使用传入的参数，并增加 zorder)
    ax.plot(rect_x, rect_z, 'r--', linewidth=2, label='Mesh Limit', zorder=10)
    
    # --- 重要：在 clear 后重新添加图例 ---
    ax.legend(loc='upper right')
    
    ax.set_title(f"Time: {tsave[frame]:.2f} | Total Energy: {total_E:.2e}")
    ax.set_xlabel('kx')
    ax.set_ylabel('kz')
    # 固定坐标轴范围，防止抖动
    ax.set_xlim(kx.min(), kx.max())
    ax.set_ylim(kz.min(), kz.max())

# --- 3. 在 FuncAnimation 中使用 fargs 传递参数 ---
# 注意 fargs 需要是一个元组
anim = FuncAnimation(fig, update, frames=traj_PSD.shape[2], interval=100,
                     fargs=box_coords) # 将 box_coords 解包传给 rect_x, rect_z

# 保存为 gif
anim.save('psd_evolution.gif', writer='pillow', fps=10)
plt.show()

traj_v = traj[0 : params.nx * params.ny * params.nz, :].reshape((params.nx, params.ny, params.nz, -1))
traj_eta = traj[params.nx * params.ny * params.nz : , :].reshape((params.nx, params.ny, params.nz, -1))

# for t_check in params.t_check_list_POD:

#     idx_sample = int(t_check / (params.dt * params.nsave))
#     v_slice = traj_v[:, :, :, idx_sample]
#     eta_slice = traj_eta[:, :, :, idx_sample]
#     idx_y_check = np.argmin(np.abs(params.y - params.y_check))
#     v_slice_ycheck = v_slice[:, idx_y_check, :]
#     eta_slice_ycheck = eta_slice[:, idx_y_check, :]

#     v_min = np.min(v_slice_ycheck)
#     v_max = np.max(v_slice_ycheck)
#     # v_spacing = 1e-6  # 等高线间距
    
#     eta_min = np.min(eta_slice_ycheck)
#     eta_max = np.max(eta_slice_ycheck)
#     # eta_spacing = 1e-6  # 等高线间距
    
#     v_dense, x_dense, z_dense = func_plot.spectral_resample(v_slice_ycheck, params.x, params.z, fom, zoom_factor=4)

#     # 构造等高线 levels
#     # levels = np.arange(v_min - v_spacing, v_max + v_spacing, v_spacing)
#     plt.figure(figsize=(10,6))
#     # plt.contourf(x, z, v_slice_ycheck.T, levels=levels, cmap='jet')
#     plt.pcolormesh(x_dense, z_dense, v_dense.T, cmap='bwr')
#     plt.colorbar()
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$z$")
#     plt.xlim(np.min(x_dense), np.max(x_dense))
#     plt.ylim(np.min(z_dense), np.max(z_dense))
#     plt.title(f"Normal velocity v at t={t_check}, y={params.y_check}")
#     plt.tight_layout()
#     plt.show()
    
#         # 1. 计算最大绝对值，确定画图范围
#     v_max = np.max(np.abs(v_dense))

#     # 2. 设置阈值 (threshold)
#     # 只有超过这个值的波动才会被画出来
#     # 比如忽略掉最大值的 5% 以下的波动（根据你的数据噪声大小调整这个 0.05）
#     threshold = 0.05 * v_max 

#     # 3. 生成两段 Levels：负数段 和 正数段，中间空开
#     # 负数段：从 -v_max 到 -threshold
#     levels_neg = np.linspace(-v_max, -threshold, 10) 
#     # 正数段：从 threshold 到 v_max
#     levels_pos = np.linspace(threshold, v_max, 10)
#     # 合并
#     custom_levels = np.concatenate([levels_neg, levels_pos])
    
#     plt.figure(figsize=(10, 6))
#     # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
#     cs = plt.contour(x_dense, z_dense, v_dense.T, levels=custom_levels, colors='black', linewidths=0.6)
#     # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
#     # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
#     # plt.colorbar()
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$z$")
#     plt.xlim(np.min(params.x), np.max(params.x))
#     plt.ylim(np.min(params.z), np.max(params.z))
#     plt.title(f"Contours of normal velocity v at t={t_check}, y={params.y_check}")
#     plt.tight_layout()
#     plt.show()

# endregion

# region 3: generate the trajectory template (if it's the first run, only using the data from the benchmark trajectory)

# We next find the trajectory template, specified in the form of 
# u_tmp = [v_tmp, eta_tmp] = f_tmp(y)cos(2pi x/Lx), where f_tmp(y) = [f_v(y); f_eta(y)] is to be determined

# The optimal (f_v, f_eta) is the solution to the following optimization problem:

# J(f_v, f_eta) = max_{f_v, f_eta} [f_v^T, f_eta^T] K [f_v; f_eta]

# satisfying the normalization constraint

# [f_v^T, f_eta^T] M [f_v; f_eta] = 1

# where M is the mass matrix associated with the inner product of (v, eta) that describes the disturbance kinetic energy of the 3D velocity field

# M = [M_v, 0; 0, M_eta], where
# M_v = L_x^2/(16pi^2)*(D1y^T@diag(Clenshaw_Curtis_weights)@D1y + (4pi^2/L_x^2) * diag(Clenshaw_Curtis_weights))
# M_eta = L_x^2/(16pi^2)*diag(Clenshaw_Curtis_weights)

# and K is the kernel matrix defined as

# K = \sum_{m = 0}^{N_snapshots - 1} h_m @ conj(h_m^T), where h_m is a vector of dimension 2 * ny defined as
# h_m = [h_v_m; h_eta_m], where
# h_v_m(y) = (D1y^T @ diag(Clenshaw_Curtis_weights) @ D1y + (4pi^2/L_x^2) * diag(Clenshaw_Curtis_weights)) @ traj_v_breve_k_neg_1_m_0(t_m)
# h_eta_m(y) = diag(Clenshaw_Curtis_weights) @ traj_eta_breve_k_neg_1_m_0(t_m)

K = np.zeros((2 * params.ny, 2 * params.ny), dtype=complex)
M = np.zeros((2 * params.ny, 2 * params.ny))

Kvv = ((fom.D1).T @ np.diag(fom.Clenshaw_Curtis_weights) @ fom.D1 + (4 * np.pi**2 / params.Lx**2) * np.diag(fom.Clenshaw_Curtis_weights))
Ketaeta = np.diag(fom.Clenshaw_Curtis_weights)

M[:params.ny, :params.ny]  = (params.Lx**2 / (16 * np.pi**2)) * Kvv
M[params.ny:, params.ny:] = (params.Lx**2 / (16 * np.pi**2)) * Ketaeta

for idx_time in range (len(params.tsave)):
    traj_v_breve_k_neg_1_m_0 = fom.FFT_2D(traj[0 : params.nx * params.ny * params.nz, idx_time].reshape((params.nx, params.ny, params.nz)))[int(params.nx/2) - 1, :, int(params.nz/2)]
    traj_eta_breve_k_neg_1_m_0 = fom.FFT_2D(traj[params.nx * params.ny * params.nz : , idx_time].reshape((params.nx, params.ny, params.nz)))[int(params.nx/2) - 1, :, int(params.nz/2)]
    h_v = Kvv @ traj_v_breve_k_neg_1_m_0
    h_eta = Ketaeta @ traj_eta_breve_k_neg_1_m_0
    h_m = np.concatenate((h_v, h_eta))
    K += np.outer(h_m, np.conj(h_m))
    
K = np.real(K) # K is a Hermitian matrix, and we want to optimize f^T @ K @ f, where f is a real-valued vector, so we can just take the real part of K.

evals, evecs = scipy.linalg.eigh(K, M, subset_by_index = [2 * params.ny - 1, 2 * params.ny - 1])
f_opt = evecs[:, -1] # the optimal f = [f_v; f_eta], notice that since we are doing linearized 3D NS, and that initially the z-mean (i.e., the zeroth Fourier mode in z direction) of v is 0, so f_v (which comes from v_breve_k_neg_1_m_0 = 0) is also 0

plt.figure()
plt.plot(params.y, f_opt[:params.ny], label="f_v(y)")
plt.plot(params.y, f_opt[params.ny:], label="f_eta(y)")
plt.title("Optimal template profile - wall-normal velocity component")
plt.xlabel("y")
plt.ylabel("f")
plt.legend()
plt.tight_layout()
plt.show()

v_template = np.cos(2 * np.pi * params.x[:, np.newaxis, np.newaxis] / params.Lx) * f_opt[np.newaxis, :params.ny, np.newaxis] * np.ones((1, 1, params.nz))
eta_template = np.cos(2 * np.pi * params.x[:, np.newaxis, np.newaxis] / params.Lx) * f_opt[np.newaxis, params.ny:, np.newaxis] * np.ones((1, 1, params.nz))
v_template_dx = -(2 * np.pi / params.Lx) * np.sin(2 * np.pi * params.x[:, np.newaxis, np.newaxis] / params.Lx) * f_opt[np.newaxis, :params.ny, np.newaxis] * np.ones((1, 1, params.nz))
eta_template_dx = -(2 * np.pi / params.Lx) * np.sin(2 * np.pi * params.x[:, np.newaxis, np.newaxis] / params.Lx) * f_opt[np.newaxis, params.ny:, np.newaxis] * np.ones((1, 1, params.nz))
v_template_dxx = -((2 * np.pi / params.Lx)**2) * np.cos(2 * np.pi * params.x[:, np.newaxis, np.newaxis] / params.Lx) * f_opt[np.newaxis, :params.ny, np.newaxis] * np.ones((1, 1, params.nz))
eta_template_dxx = -((2 * np.pi / params.Lx)**2) * np.cos(2 * np.pi * params.x[:, np.newaxis, np.newaxis] / params.Lx) * f_opt[np.newaxis, params.ny:, np.newaxis] * np.ones((1, 1, params.nz))

traj_template = np.concatenate((v_template.ravel(), eta_template.ravel()))
traj_template_dx = np.concatenate((v_template_dx.ravel(), eta_template_dx.ravel()))
traj_template_dxx = np.concatenate((v_template_dxx.ravel(), eta_template_dxx.ravel()))
np.save(params.fname_traj_template, traj_template)
np.save(params.fname_traj_template_dx, traj_template_dx)
np.save(params.fname_traj_template_dxx, traj_template_dxx)

# endregion
