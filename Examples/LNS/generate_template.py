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

### First, we try to set up the initial localized disturbance

Lx = 32
Ly = 2 # from -1 to 1
Lz = 16

nx = 64
ny = 65 # ny includes the boundary points when using Chebyshev grid
nz = 64

x = np.linspace(0, Lx, num=nx, endpoint=False)
y = np.cos(np.pi * np.linspace(0, ny - 1, num=ny) / (ny - 1))  # Chebyshev grid in y direction, location from 1 to -1
z = np.linspace(0, Lz, num=nz, endpoint=False)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

Re = 6000
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

amp = 1.0

psi0 = amp * (1-Y**2)**2 * ((X - Lx/2)/2) * (Z - Lz/2) ** 2 * np.exp(-((X - Lx/2)/2)**2 - ((Z - Lz/2)/2)**2)
v0 = fom.diff_z(psi0, order = 1)
eta0 = fom.diff_x(fom.diff_1_y(psi0), order = 1)
  
# endregion

# region 2: Simulations

# Before doing any data generation, we first run on the benchmark training trajectory to get the trajectory template

print("Running the benchmark simulation")
traj_init = np.concatenate((v0.ravel(), eta0.ravel()))
traj, tsave = tstep_kse_fom.time_step(traj_init, nsave) # traj is in the physical domain and is of the shape (2 * nx * ny * nz, nsave_samples)
traj_v = traj[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
traj_eta = traj[nx * ny * nz : , :].reshape((nx, ny, nz, -1))

t_check_list = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]
y_check = -0.56

for t_check in t_check_list:

    idx_sample = int(t_check / (dt * nsave))
    v_slice = traj_v[:, :, :, idx_sample]
    eta_slice = traj_eta[:, :, :, idx_sample]
    idx_y_check = np.argmin(np.abs(y - y_check))
    v_slice_ycheck = v_slice[:, idx_y_check, :]
    eta_slice_ycheck = eta_slice[:, idx_y_check, :]

    v_min = np.min(v_slice_ycheck)
    v_max = np.max(v_slice_ycheck)
    # v_spacing = 1e-6  # 等高线间距
    
    eta_min = np.min(eta_slice_ycheck)
    eta_max = np.max(eta_slice_ycheck)
    # eta_spacing = 1e-6  # 等高线间距

    # 构造等高线 levels
    # levels = np.arange(v_min - v_spacing, v_max + v_spacing, v_spacing)
    plt.figure(figsize=(10,6))
    # plt.contourf(x, z, v_slice_ycheck.T, levels=levels, cmap='jet')
    plt.pcolormesh(x, z, v_slice_ycheck.T, cmap='bwr')
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$z$")
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(z), np.max(z))
    plt.title(f"Normal velocity v at t={t_check}, y={y_check}")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
    cs = plt.contour(x, z, v_slice_ycheck.T, colors='black', linewidths=0.6)
    # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
    # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
    # plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$z$")
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(z), np.max(z))
    plt.title(f"Contours of normal velocity v at t={t_check}, y={y_check}")
    plt.tight_layout()
    plt.show()

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

K = np.zeros((2 * ny, 2 * ny), dtype=complex)
M = np.zeros((2 * ny, 2 * ny))

Kvv = ((fom.D1).T @ np.diag(fom.Clenshaw_Curtis_weights) @ fom.D1 + (4 * np.pi**2 / Lx**2) * np.diag(fom.Clenshaw_Curtis_weights))
Ketaeta = np.diag(fom.Clenshaw_Curtis_weights)

M[:ny, :ny]  = (Lx**2 / (16 * np.pi**2)) * Kvv
M[ny:, ny:] = (Lx**2 / (16 * np.pi**2)) * Ketaeta

for idx_time in range (len(tsave)):
    traj_v_breve_k_neg_1_m_0 = fom.FFT_2D(traj[0 : nx * ny * nz, idx_time].reshape((nx, ny, nz)))[int(nx/2) - 1, :, int(nz/2)]
    traj_eta_breve_k_neg_1_m_0 = fom.FFT_2D(traj[nx * ny * nz : , idx_time].reshape((nx, ny, nz)))[int(nx/2) - 1, :, int(nz/2)]
    h_v = Kvv @ traj_v_breve_k_neg_1_m_0
    h_eta = Ketaeta @ traj_eta_breve_k_neg_1_m_0
    h_m = np.concatenate((h_v, h_eta))
    K += np.outer(h_m, np.conj(h_m))
    
K = np.real(K) # K is a Hermitian matrix, and we want to optimize f^T @ K @ f, where f is a real-valued vector, so we can just take the real part of K.

evals, evecs = scipy.linalg.eigh(K, M, subset_by_index = [2 * ny - 1, 2 * ny - 1])
f_opt = evecs[:, -1] # the optimal f = [f_v; f_eta], notice that since we are doing linearized 3D NS, and that initially the z-mean (i.e., the zeroth Fourier mode in z direction) of v is 0, so f_v (which comes from v_breve_k_neg_1_m_0 = 0) is also 0

plt.figure()
plt.plot(y, f_opt[:ny], label="f_v(y)")
plt.plot(y, f_opt[ny:], label="f_eta(y)")
plt.title("Optimal template profile - wall-normal velocity component")
plt.xlabel("y")
plt.ylabel("f")
plt.legend()
plt.tight_layout()
plt.show()

traj_template = np.cos(2 * np.pi * x[:, np.newaxis, np.newaxis] / Lx) * f_opt[np.newaxis, :, np.newaxis] * np.ones((1, 1, nz))
traj_template_dx = fom.diff_x(traj_template, order = 1)
traj_template_dxx = fom.diff_x(traj_template, order = 2)
traj_template = traj_template.ravel()
traj_template_dx = traj_template_dx.ravel()
traj_template_dxx = traj_template_dxx.ravel()
np.save(fname_traj_template, traj_template)
np.save(fname_traj_template_dx, traj_template_dx)
np.save(fname_traj_template_dxx, traj_template_dxx)

# endregion
