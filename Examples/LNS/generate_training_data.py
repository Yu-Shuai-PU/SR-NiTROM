import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys
import os
# plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
# plt.rc('text.latex',preamble=r'\usepackage{amsmath}')
sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

import classes
import fom_class_LNS
from fom_class_LNS import Chebyshev_diff_mat as diff_y_mat
from fom_class_LNS import Chebyshev_diff_FFT as diff_y

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

fom = fom_class_LNS.LNS(Lx, Ly, Lz, nx, ny, nz, y, Re, U_base, U_base_dy, U_base_dyy)
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
psi0[:, :, :, 0] = amp * (1-Y**2)**2 * ((X - Lx/2)/2) * (Z - Lz/2) * np.exp(-((X - Lx/2)/2)**2 - ((Z - Lz/2)/2)**2)

for k in range (n_traj):
    v0[:, :, :, k] = fom.diff_z_input_3D(psi0[:, :, :, k], order = 1)
    eta0[:, :, :, k] = fom.diff_x_input_3D(fom.diff_1_y_input_3D(psi0[:, :, :, k]), order = 1)
  
# endregion

# region 2: Simulations

pool_inputs = (MPI.COMM_WORLD, n_traj)
pool = classes.mpi_pool(*pool_inputs)

# Before doing any data generation, we first run on the benchmark training trajectory to get the trajectory template

for k in range (pool.my_n_traj):
    traj_idx = k + pool.disps[pool.rank]
    print("Running simulation %d/%d"%(traj_idx,n_traj - 1))
    q0 = np.concatenate((v0[:, :, :, traj_idx].ravel(), eta0[:, :, :, traj_idx].ravel()))
    
    traj, tsave = tstep_kse_fom.time_step(q0, nsave)
    np.save(fname_traj_init%traj_idx,q0)
    np.save(fname_time, tsave)
    np.save(fname_traj % (traj_idx), traj)
   
    
    
    # traj_v = traj[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
    # traj_eta = traj[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
    
    # t_check_list = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]
    # y_check = -0.56

    # for t_check in t_check_list:

    #     idx_sample = int(t_check / (dt * nsave))
    #     v_slice = traj_v[:, :, :, idx_sample]
    #     eta_slice = traj_eta[:, :, :, idx_sample]
    #     idx_y_check = np.argmin(np.abs(y - y_check))
    #     v_slice_ycheck = v_slice[:, idx_y_check, :]
    #     eta_slice_ycheck = eta_slice[:, idx_y_check, :]

    #     v_min = np.min(v_slice_ycheck)
    #     v_max = np.max(v_slice_ycheck)
    #     # v_spacing = 1e-6  # 等高线间距
        
    #     eta_min = np.min(eta_slice_ycheck)
    #     eta_max = np.max(eta_slice_ycheck)
    #     # eta_spacing = 1e-6  # 等高线间距

    #     # 构造等高线 levels
    #     # levels = np.arange(v_min - v_spacing, v_max + v_spacing, v_spacing)
    #     plt.figure(figsize=(10,6))
    #     # plt.contourf(x, z, v_slice_ycheck.T, levels=levels, cmap='jet')
    #     plt.pcolormesh(x, z, v_slice_ycheck.T, cmap='bwr')
    #     plt.colorbar()
    #     plt.xlabel(r"$x$")
    #     plt.ylabel(r"$z$")
    #     plt.xlim(np.min(x), np.max(x))
    #     plt.ylim(np.min(z), np.max(z))
    #     plt.title(f"Normal velocity v at t={t_check}, y={y_check}")
    #     plt.tight_layout()
    #     plt.show()
        
    #     plt.figure(figsize=(10, 6))
    #     # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
    #     cs = plt.contour(x, z, v_slice_ycheck.T, colors='black', linewidths=0.6)
    #     # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
    #     # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
    #     # plt.colorbar()
    #     plt.xlabel(r"$x$")
    #     plt.ylabel(r"$z$")
    #     plt.xlim(np.min(x), np.max(x))
    #     plt.ylim(np.min(z), np.max(z))
    #     plt.title(f"Contours of normal velocity v at t={t_check}, y={y_check}")
    #     plt.tight_layout()
    #     plt.show()
        
    ## save the data
    # np.save()
    
... 

    
# Now, we have the trajectory data:
# sol_v: shape (nx, ny, nz, nt_samples)
# sol_eta: shape (nx, ny, nz, nt_samples)

# We next find the trajectory template, specified in the form of 
# u_tmp = [v_tmp, eta_tmp] = f_tmp(y)cos(2pi x/Lx), where f_tmp(y) = [f_v(y); f_eta(y)] is to be determined

# 1. Compute the giant kernel function

# K(y', y) = \sum_{m = 0}^{Nsamples - 1} (\frac{1}{2Lz}\int_0^{Lz} conjugate{\tilde{u}_{-1}}(y', z, t_m)} dz) (\frac{1}{2Lz}\int_0^{Lz} \tilde{u}_{-1}(y, z, t_m)} dz)^T,

# where \tilde{u}_{-1} = [\tilde{v}_{-1}; \tilde{eta}_{-1}] is the inverse Fourier transform of u_hat at k = -1, where

# [v, eta](x, y, z, t) = \sum_{kx = - nx/2}^{nx/2 - 1} [\tilde{v}_k, \tilde{eta}_k](y, z, t) exp(2j * pi * kx * x / Lx)

# 2. Take the real part of the kernel function

# 3. Solve the eigenvalue problem to get f_tmp(y)

# 1. Compute the \tilde{u}_{-1} = [\tilde{v}_{-1}; \tilde{eta}_{-1}] and the kernel matrix K
v_hat_k_neg_1 = np.zeros((ny, nz), dtype=complex)
eta_hat_k_neg_1 = np.zeros((ny, nz), dtype=complex)

K = np.zeros((2 * ny, 2 * ny), dtype=complex)

for idx_time in range (len(tsave)):
    v_hat_k_neg_1 = fom.Fourier_transform_x(sol_v[:, :, :, idx_time])[int(nx/2) - 1, :, :]
    eta_hat_k_neg_1 = fom.Fourier_transform_x(sol_eta[:, :, :, idx_time])[int(nx/2) - 1, :, :]
    
    K[:ny, :ny] += (0.5 ** 2) * np.outer(
                    np.conj(np.mean(v_hat_k_neg_1, axis=1)), # average over z direction
                    np.mean(v_hat_k_neg_1, axis=1))
    K[:ny, ny:] += (0.5 ** 2) * np.outer(
                    np.conj(np.mean(v_hat_k_neg_1, axis=1)),
                    np.mean(eta_hat_k_neg_1, axis=1))
    K[ny:, :ny] += (0.5 ** 2) * np.outer(
                    np.conj(np.mean(eta_hat_k_neg_1, axis=1)),
                    np.mean(v_hat_k_neg_1, axis=1))
    K[ny:, ny:] += (0.5 ** 2) * np.outer(
                    np.conj(np.mean(eta_hat_k_neg_1, axis=1)),
                    np.mean(eta_hat_k_neg_1, axis=1))
    
weights = fom.Clenshaw_Curtis_weights
    
# 2. Compute the primary eigenvector of the real part of the kernal matrix K with Clenshaw-Curtis quadrature weights

W = np.zeros((2 * ny, 2 * ny))
W[:ny, :ny] = np.diag(np.sqrt(weights))
W[ny:, ny:] = np.diag(np.sqrt(weights))

W_inv = np.zeros((2 * ny, 2 * ny))
W_inv[:ny, :ny] = np.diag(1.0 / np.sqrt(weights))
W_inv[ny:, ny:] = np.diag(1.0 / np.sqrt(weights))

eigvals, eigvecs = np.linalg.eigh(W @ np.real(K) @ W)
u_template_profile = W_inv @ eigvecs[:, np.argmax(eigvals)]

u_template = np.zeros((2 * nx * ny * nz))
u_template_dx = np.zeros((2 * nx * ny * nz))
u_template_quarter_shifted = np.zeros((2 * nx * ny * nz))
for idx_x in range (nx):
    for idx_y in range (ny):
        for idx_z in range (nz):
            u_template[idx_x * ny * nz + idx_y * nz + idx_z] = u_template_profile[idx_y] * np.cos(2 * np.pi * x[idx_x] / Lx)
            u_template[nx * ny * nz + idx_x * ny * nz + idx_y * nz + idx_z] = u_template_profile[ny + idx_y] * np.cos(2 * np.pi * x[idx_x] / Lx)
            u_template_quarter_shifted[idx_x * ny * nz + idx_y * nz + idx_z] = u_template_profile[idx_y] * np.sin(2 * np.pi * x[idx_x] / Lx)
            u_template_quarter_shifted[nx * ny * nz + idx_x * ny * nz + idx_y * nz + idx_z] = u_template_profile[ny + idx_y] * np.sin(2 * np.pi * x[idx_x] / Lx)
            u_template_dx[idx_x * ny * nz + idx_y * nz + idx_z] = - (2 * np.pi / Lx) * u_template_profile[idx_y] * np.sin(2 * np.pi * x[idx_x] / Lx)
            u_template_dx[nx * ny * nz + idx_x * ny * nz + idx_y * nz + idx_z] = - (2 * np.pi / Lx) * u_template_profile[ny + idx_y] * np.sin(2 * np.pi * x[idx_x] / Lx)
# u_template[0 : nx * ny * nz] = (u_template_profile[:ny][:, np.newaxis] * np.cos(2 * np.pi * x[:, np.newaxis] / Lx)).reshape((nx * ny * nz,))
# u_template[nx * ny * nz : ] = (u_template_profile[ny:][:, np.newaxis] * np.cos(2 * np.pi * x[:, np.newaxis] / Lx)).reshape((nx * ny * nz,))

print("inner product of u_template: ", fom.inner_product_3D(u_template[0 : nx * ny * nz].reshape((nx, ny, nz)), u_template[0 : nx * ny * nz].reshape((nx, ny, nz))) + fom.inner_product_3D(u_template[nx * ny * nz : ].reshape((nx, ny, nz)), u_template[nx * ny * nz : ].reshape((nx, ny, nz))))
print("inner product of u_template_dx: ", fom.inner_product_3D(u_template_dx[0 : nx * ny * nz].reshape((nx, ny, nz)), u_template_dx[0 : nx * ny * nz].reshape((nx, ny, nz))) + fom.inner_product_3D(u_template_dx[nx * ny * nz : ].reshape((nx, ny, nz)), u_template_dx[nx * ny * nz : ].reshape((nx, ny, nz))))
plt.figure()
plt.plot(y, u_template_profile[:ny], label="f_v(y)")
plt.show()
plt.figure()
plt.plot(y, u_template_profile[ny:], label="f_eta(y)")
plt.show()
plt.close('all')

fom.load_template(u_template, u_template_dx)

# 3. Check the denominator of the reconstruction equation
# First of all, we eliminate the shifting amount to zero
# the shift amount is defined as 
# c(t) = arg(<u, u_template> + 1j * <u, u_template_quarter_shifted>) * (Lx / (2 * pi))

denom = np.zeros((len(tsave)))
denom_reference = np.zeros((len(tsave)))
shift_amount = np.zeros((len(tsave)))
for idx_time in range (len(tsave)):
    v_current = traj[0 : nx * ny * nz, idx_time].reshape((nx, ny, nz))
    eta_current = traj[nx * ny * nz : , idx_time].reshape((nx, ny, nz))
    
    
    
    
    ip_u_u_template = fom.inner_product_3D(v_current, u_template[0 : nx * ny * nz].reshape((nx, ny, nz))) + fom.inner_product_3D(eta_current, u_template[nx * ny * nz : ].reshape((nx, ny, nz)))
    ip_u_u_template_quarter_shifted = fom.inner_product_3D(v_current, u_template_quarter_shifted[0 : nx * ny * nz].reshape((nx, ny, nz))) + fom.inner_product_3D(eta_current, u_template_quarter_shifted[nx * ny * nz : ].reshape((nx, ny, nz)))
    shift_amount[idx_time] = np.angle(ip_u_u_template + 1j * ip_u_u_template_quarter_shifted) * (Lx / (2 * np.pi))
    
    u_shifted = fom.shift(traj[:, idx_time], -shift_amount[idx_time])
    v_shifted_dx = fom.Fourier_diff_x_3D(u_shifted[0 : nx * ny * nz].reshape((nx, ny, nz)), order = 1)
    eta_shifted_dx = fom.Fourier_diff_x_3D(u_shifted[nx * ny * nz : ].reshape((nx, ny, nz)), order = 1)
    u_shifted_dx = np.concatenate((v_shifted_dx.ravel(), eta_shifted_dx.ravel()))
    denom[idx_time] = fom.evaluate_fom_shift_speed_denom(u_shifted_dx)
    denom_reference[idx_time] = fom.inner_product_3D(v_shifted_dx, v_shifted_dx) + fom.inner_product_3D(eta_shifted_dx, eta_shifted_dx)
    
plt.figure()
plt.plot(tsave, denom)
plt.xlabel("Time")
plt.ylabel("Denominator of shift speed evaluation")
plt.title("Denominator of shift speed evaluation over time")
plt.tight_layout()
plt.show()





