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
from localized disturbances in wall-bounded shear flows"

See Schmid and Henningson 2001, "Stability and Transition in Shear Flows":
    a) p.147 and p.148 Fig. 4.19 for the time evolution (and downstream drifting) of initial localized disturbance.
    b) p.144 Fig. 4.16(c)(f) for the shape of initial localized disturbance.
    c) Fig.1 and eq.1(a-c) and eq.2 from Henningson et al. (1993) for the mathematical form of the initial localized disturbance
    d) Table 1 from Henningson et al. (1993) for the domain size, grid points number, and other parameters used in the simulation
"""

# region 1: Initialization

### First, we try to set up the initial localized disturbance

Lx = 48
Ly = 2 # from -1 to 1
Lz = 24

nx = 96
ny = 65 # ny includes the boundary points when using Chebyshev grid
nz = 96

Re = 3000

T = 40
dt = 5.0
dt_sample = 5.0
nt = int(T/dt) + 1
sample_interval = int(dt_sample / dt)
nt_samples = int(nt / sample_interval) + 1

small_amp  = 0.0001
medium_amp = 0.0699
large_amp  = 0.1398

amp = small_amp

x = np.linspace(-10, 38, num=nx, endpoint=False)
y = np.cos(np.pi * np.linspace(0, ny - 1, num=ny) / (ny - 1))  # Chebyshev grid in y direction, location from 1 to -1
z = np.linspace(-Lz/2, Lz/2, num=nz, endpoint=False)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
psi = amp * (1-Y**2)**2 * (X/2) * Z * np.exp(-(X/2)**2 - (Z/2)**2)

# Define the base flow
U_base = 1 - y**2
U_base_dy = -2 * y
U_base_dyy = -2 * np.ones_like(y)

fom = fom_class_LNS.LNS(Lx, Ly, Lz, nx, ny, nz, y, Re, U_base, U_base_dy, U_base_dyy)

# Pick normal velocity and vorticity as the primitive variables
v_init   = np.zeros_like(psi)
psi_y    = np.zeros_like(psi)
eta_init = np.zeros_like(psi)

for idx_x in range (nx):
    for idx_y in range (ny):
        psi_slice_z = psi[idx_x, idx_y, :]
        v_init[idx_x, idx_y, :] = fom.Fourier_diff_z(psi_slice_z, order = 1)

for idx_x in range (nx):
    for idx_z in range (nz):
        psi_slice = psi[idx_x, :, idx_z]
        psi_y[idx_x, :, idx_z] = diff_y(psi_slice)
        
for idx_y in range (ny):
    for idx_z in range (nz):
        psi_y_slice_x = psi_y[:, idx_y, idx_z]
        eta_init[:, idx_y, idx_z] = fom.Fourier_diff_x(psi_y_slice_x, order = 1)

# endregion

# region 2: Obtain the analytical form of solution at t = 30

### 1. Perform double Fourier transforms in x and z directions; 
### for each (kx, kz) mode, construct the initial value problem of ODE via Chebyshev diff matrix
v_hat = fom.Fourier_transform_xz(v_init) # shape: vhat(kx, y, kz) has shape (nx, ny, nz)
eta_hat = fom.Fourier_transform_xz(eta_init)
sol_v_hat = np.zeros((nx, ny, nz, nt_samples), dtype=complex)
sol_eta_hat = np.zeros((nx, ny, nz, nt_samples), dtype=complex)
state = np.zeros((2 * (ny - 2)), dtype=complex)

### To complete the problem, we need to enforce the BCs on the differentiation matrices
D1 = diff_y_mat(y)
D2, D4 = fom.apply_BC(D1) # Both are of shape (ny-2, ny - 2)

Id = np.eye(ny - 2, dtype=complex)
U_base_mat = np.diag(U_base[1:-1])
U_base_dy_mat = np.diag(U_base_dy[1:-1])
U_base_dyy_mat = np.diag(U_base_dyy[1:-1])
M = np.zeros((2 * (ny - 2), 2 * (ny - 2)), dtype=complex)
L = np.zeros((2 * (ny - 2), 2 * (ny - 2)), dtype=complex)

# q_init = np.concatenate((v_init.ravel(), eta_init.ravel()))
# rhs_2 = fom.linear(q_init) # just for testing

# rhs_v_hat = np.zeros((nx, ny, nz), dtype=complex)
# rhs_eta_hat = np.zeros((nx, ny, nz), dtype=complex)
# rhs_1_hat = np.zeros(2 * nx * ny * nz, dtype=complex)
# rhs_1 = np.zeros(2 * nx * ny * nz)

for idx_kx in range (nx):
    kx = fom.kx[idx_kx]
    for idx_kz in range (nz):
        kz = fom.kz[idx_kz]
        Laplace = D2 - (kx**2 + kz**2) * Id
        Bi_Laplace = D4 - 2 * (kx**2 + kz**2) * D2 + (kx**2 + kz**2)**2 * Id 
        M[:ny - 2, :ny - 2] = Laplace
        M[ny - 2:, ny - 2:] = Id
        L[:ny - 2, :ny - 2] = -1j * kx * (U_base_mat @ Laplace) + 1j * kx * U_base_dyy_mat + Bi_Laplace / Re
        L[ny - 2:, :ny - 2] = -1j * kz * U_base_dy_mat
        L[ny - 2:, ny - 2:] = -1j * kx * U_base_mat + Laplace / Re
        linear_mat = scipy.linalg.solve(M, L)
        counter = 0
        sol_v_hat[idx_kx, 0, idx_kz, :] = 0.0
        sol_v_hat[idx_kx, -1, idx_kz, :] = 0.0
        sol_eta_hat[idx_kx, 0, idx_kz, :] = 0.0
        sol_eta_hat[idx_kx, -1, idx_kz, :] = 0.0
        state[:ny-2] = v_hat[idx_kx, 1:-1, idx_kz]
        state[ny-2:] = eta_hat[idx_kx, 1:-1, idx_kz]
        # Compute the exponential matrix once for all time steps
        
        # rhs_q_slice = linear_mat @ state

        # rhs_v_hat[idx_kx, 1:-1, idx_kz] = rhs_q_slice[:ny-2]
        # rhs_eta_hat[idx_kx, 1:-1, idx_kz] = rhs_q_slice[ny-2:]

        exp_linear_mat = scipy.linalg.expm(linear_mat * dt)
        for idx_time in range (nt):
            t = idx_time * dt
            if idx_time % sample_interval == 0:
                print(f"Processing kx={kx}, kz={kz}, t={t:.3f}")
                sol_v_hat[idx_kx, 1:-1, idx_kz, counter] = state[0:ny-2]
                sol_eta_hat[idx_kx, 1:-1, idx_kz, counter] = state[ny-2:]
                counter += 1
            state = exp_linear_mat @ state
        
        """
        # Method 2: Solve the eigenvalue problem and reconstruct the solution
        # Note: This method currently has issues with matching the adjoint modes to the direct modes
        ################################################################################
        ################################################################################
        ################################################################################
        # eig_values, eig_vectors = scipy.linalg.eig(L, M)
        # omega = 1j * eig_values
        # growth_rates = omega.imag
        # sort_indices = np.argsort(-growth_rates)
        # eig_values_sorted = eig_values[sort_indices]
        # omega_sorted = omega[sort_indices]
        # eig_vectors_sorted = eig_vectors[:, sort_indices] # omega is sorted from most unstable to most stable mode
        
        # # Next, solve the adjoint problem to get the coefficients
        # L_adj = L.conj().T
        # M_adj = M.conj().T
        # eig_values_adj, eig_vectors_adj = scipy.linalg.eig(L_adj, M_adj)
        # omega_adj = 1j * eig_values_adj
        # omega_adj_matched = np.zeros_like(omega_adj)
        # is_matched = np.zeros(len(omega_adj), dtype=bool)

        # threshold = 1e-5 # 设定的绝对值误差容忍度

        # for i, current_omega in enumerate(omega_sorted):
        #     # 计算当前 omega 与所有未匹配的 omega_adj 之间的绝对误差
        #     error = np.abs(current_omega - (-omega_adj.conj()))

        #     # 排除已匹配的
        #     error[is_matched] = np.inf

        #     # 找到误差最小的索引
        #     min_error_index = np.argmin(error)
        #     min_error = error[min_error_index]

        #     if min_error < threshold:
        #         # 如果最小误差在容忍度内，则认为匹配成功
        #         omega_adj_matched[i] = omega_adj[min_error_index]
        #         is_matched[min_error_index] = True
        #         # 匹配成功后，你可能还需要将 eig_vectors_adj 的列进行对应排序
        #         # eig_vectors_adj_matched[:, i] = eig_vectors_adj[:, min_error_index]
        #     else:
        #         # 如果没有找到足够接近的，可能是数值误差太大或有不匹配的情况
        #         # 你需要决定在这种情况下如何处理，例如保留为零或标记为未找到
        #         raise ValueError(f"无法为 omega_sorted[{i}] = {current_omega:.6g} 找到匹配的 omega_adj。最小误差: {min_error:.6g}")
        #         # print(f"警告：无法为 omega_sorted[{i}] = {current_omega:.6g} 找到匹配的 omega_adj。最小误差: {min_error:.6g}")

        # eig_vectors_adj_sorted = eig_vectors_adj[:, np.argsort(omega_adj_matched)]
        
        # print(f"Total misalignment error in omega: {np.sum(np.abs(omega_sorted - (-omega_adj_matched.conj()))):.6g}")
        
        # # 尝试画第一个特征向量和伴随特征向量
        # idx = 6
        # plt.figure()
        # plt.plot(y[1:-1], np.abs(eig_vectors_sorted[0:ny-2,idx]), label="Direct v mode")
        # plt.plot(y[1:-1], np.abs(eig_vectors_sorted[ny-2:,idx]), label="Direct eta mode")
        # plt.plot(y[1:-1], np.abs(eig_vectors_adj_sorted[0:ny-2,idx]), label="Adjoint v mode")
        # plt.plot(y[1:-1], np.abs(eig_vectors_adj_sorted[ny-2:,idx]), label="Adjoint eta mode")
        # plt.xlabel("y")
        # plt.ylabel("Magnitude")
        # plt.title(f"Mode shape for kx={kx}, kz={kz}")
        # plt.legend()
        # plt.show()
        # print(f"inner product: {eig_vectors_sorted[:,idx].conj().T @ M @ eig_vectors_adj_sorted[:,idx]:.6g}")
        # print(f"energy: {eig_vectors_sorted[:,idx].conj().T @ M @ eig_vectors_sorted[:,idx]:.6g}")
        # ...
        """

### Testing: to flatten the solution

# rhs_v = fom.Fourier_inverse_transform_xz(rhs_v_hat)
# rhs_eta = fom.Fourier_inverse_transform_xz(rhs_eta_hat)
# rhs_1[:nx * ny * nz] = rhs_v.ravel()
# rhs_1[nx * ny * nz :] = rhs_eta.ravel()

# print(f"Recovery error in rhs: {np.linalg.norm(rhs_1 - rhs_2):.6g}")


idx_sample = int(10.0 / dt_sample)
v_slice = fom.Fourier_inverse_transform_xz(sol_v_hat[:, :, :, idx_sample])
eta_slice = fom.Fourier_inverse_transform_xz(sol_eta_hat[:, :, :, idx_sample])
q = np.concatenate((v_slice.ravel(), eta_slice.ravel()))
v_slice_recover = q[0 : nx * ny * nz].reshape((nx, ny, nz))
eta_slice_recover = q[nx * ny * nz : ].reshape((nx, ny, nz))
print(f"Recovery error in v at t=10: {np.linalg.norm(v_slice - v_slice_recover):.6g}")
print(f"Recovery error in eta at t=10: {np.linalg.norm(eta_slice - eta_slice_recover):.6g}")


### Check the solution at t = 10, y = -0.56
t_check_list = [0.0, 10.0, 20.0, 30.0, 40.0]
y_check = -0.56




output = fom.linear

for t_check in t_check_list:

    idx_sample = int(t_check / dt_sample)
    v_slice = fom.Fourier_inverse_transform_xz(sol_v_hat[:, :, :, idx_sample])
    eta_slice = fom.Fourier_inverse_transform_xz(sol_eta_hat[:, :, :, idx_sample])
    # Next, find the index of y closest to -0.56
    idx_y_check = np.argmin(np.abs(y - y_check))
    v_slice_ycheck = v_slice[:, idx_y_check, :]
    eta_slice_ycheck = eta_slice[:, idx_y_check, :]

    v_min = np.min(v_slice_ycheck)
    v_max = np.max(v_slice_ycheck)
    v_spacing = 1e-6  # 等高线间距
    
    # eta_min = np.min(eta_slice_ycheck)
    # eta_max = np.max(eta_slice_ycheck)
    # eta_spacing = 1e-6  # 等高线间距

    # 构造等高线 levels
    levels = np.arange(v_min - v_spacing, v_max + v_spacing, v_spacing)
    plt.figure(figsize=(10,6))
    plt.contourf(x, z, v_slice_ycheck.T, levels=levels, cmap='jet')
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$z$")
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(z), np.max(z))
    plt.title(f"Normal velocity v at t={t_check}, y={y_check}")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
    # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
    plt.xlabel(r"$x$")
    plt.ylabel(r"$z$")
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(z), np.max(z))
    plt.title(f"Contours of normal velocity v at t={t_check}, y={y_check}")
    plt.tight_layout()
    plt.show()

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

