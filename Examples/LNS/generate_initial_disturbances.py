"""
This module generates initial disturbances for simulations.

Typically, we have 3 types of initial disturbances:

1. Disturbances centered around the least-stable 2D Tollmien-Schlichting (TS) wave (mainly exponential decay/growth, no kz = 0 components).
2. Axis-symmetric disturbances with 0 normal vorticity (mainly algebraic transient growth driven by the lift-up mechanism).
3. A pair of counter-rotating streamwise vortices centered around a pair of oblique waves (mainly lift-up mechanism).

"""
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
sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

import configs
import classes
import fom_class_LNS
import func_plot

def generate_IC_around_2D_TS(params, fom):
    
    # step 1: compute the least-stable 2D TS wave at given Re
    Lx = params.Lx
    nx = params.nx
    ny = params.ny
    nmodes_x = nx // 2
    
    kx_range = 2 * np.pi / Lx * np.linspace(1, nmodes_x - 1, nmodes_x - 1, dtype=int, endpoint=True)  # Feasible streamwise wavenumbers, to find the optimal wavenumber we only need positive values because of symmetry
    # kz = 0.0  # Spanwise wavenumber for 2D TS wave
    
    Id = np.eye(ny - 2, dtype=complex)
    D2 = fom.D2
    D4 = fom.D4
    
    maximal_growth_rate = -np.inf
    kx_optimal = None
    v_hat_optimal = np.zeros(ny, dtype=complex)
    
    for kx in kx_range: # formulate the 2D Orr-Sommerfeld eigenvalue problem for each kx and find the one with the largest growth rate
        Laplace = D2 - (kx**2) * Id
        Bi_Laplace = D4 - 2 * (kx**2) * D2 + (kx**2)**2 * Id 
        M = Laplace
        L = -1j * kx * (fom.U_base_mat @ Laplace) + 1j * kx * fom.U_base_dyy_mat + Bi_Laplace / fom.Re
        
        eigvals, eigvecs = scipy.linalg.eig(L, M)
        
        growth_rates = np.real(eigvals)
        idx_max_growth_rate = np.argmax(growth_rates)
        max_growth_rate = growth_rates[idx_max_growth_rate]
        
        if max_growth_rate > maximal_growth_rate:
            maximal_growth_rate = max_growth_rate
            kx_optimal = kx
            
            eigvec_optimal = eigvecs[:, idx_max_growth_rate]
            v_hat_optimal[1:-1] = eigvec_optimal
            
    return kx_optimal, v_hat_optimal, maximal_growth_rate

def main():
        
    # region 1: Initialization
    params = configs.load_configs()
    fom = fom_class_LNS.LNS(params.Lx, params.Ly, params.Lz, 
                            params.nx, params.ny, params.nz,
                            params.y, params.Re,
                            params.U_base, params.U_base_dy, params.U_base_dyy)
    
    tstep_kse_fom = fom_class_LNS.time_step_LNS(fom, params.time)
    
    # region 2: Generate initial disturbances that are centered around the least-stable 2D Tollmien-Schlichting (TS) wave at given Re

    # The initial disturbance looks like a wavepacket smoothed by Gaussian envelopes in both streamwise and spanwise directions
    # v(x, y, z) = Real( amp * vhat_TS(y) * exp(1j * alpha_TS * (x - Lx/2)) * exp(-((x - Lx/2)/sigma_x)^2) * exp(-((z - Lz/2)/sigma_z)^2) )
    # which accounts for amplitude, TS normal profile, streamwise carrier wave, and Gaussian envelopes in both x and z directions

    # Next, we figure out the u profile from continuity equation
    # du/dx + dv/dy + dw/dz = 0, considering w approximately zero for the initial disturbance centered around 2D TS wave
    # du/dx = -dv/dy
    # from here we have u(x, y, z)

    # Eventually, we compute the initial normal vorticity
    # eta(x, y, z) = du/dz - dw/dx approximately du/dz
    
    kx_TS, vhat_TS, maximal_growth_rate = generate_IC_around_2D_TS(params, fom)
    
    print("Least-stable 2D TS wave at Re = %d has streamwise wavenumber kx = %.4f, maximal growth rate = %.6f" % (params.Re, kx_TS, maximal_growth_rate))
    
    plt.figure()
    plt.plot(params.y, np.real(vhat_TS), label='Real part')
    plt.plot(params.y, np.imag(vhat_TS), label='Imaginary part')
    plt.title('Normal velocity profile of least-stable 2D TS wave at Re = %d' % params.Re)
    plt.xlabel('y')
    plt.ylabel('v_hat(y)')
    plt.legend()
    plt.grid()
    plt.show()
    
    X, Y, Z = params.X, params.Y, params.Z
    v0 = (vhat_TS[np.newaxis, :, np.newaxis] * np.exp(1j * kx_TS * (X - params.Lx / 2)) * np.exp(-((X - params.Lx / 2) / 32) ** 2) * np.exp(-((Z - params.Lz / 2) / 8) ** 2)).real
    # v0 = (vhat_TS[np.newaxis, :, np.newaxis] * np.exp(1j * kx_TS * (X - params.Lx / 2))).real
    dv0_dy = fom.diff_1_y(v0)
    dv0_dy_hat = fom.FFT_1D(dv0_dy, axis=0)
    u0_hat = np.zeros_like(dv0_dy_hat)
    for idx_x in range(params.nx):
        kx = fom.kx[idx_x]
        if kx != 0:
            u0_hat[idx_x, :, :] = 1j * dv0_dy_hat[idx_x, :, :] / kx
        else:
            u0_hat[idx_x, :, :] = 0.0
    u0 = fom.IFFT_1D(u0_hat, axis=0)
    eta0 = fom.diff_z(u0, order = 1)
    
    # compute the initial disturbance energy for normalization
    initial_disturbance_energy = fom.inner_product_3D(v0, eta0, v0, eta0)
    
    v0 = v0[:, :, :, np.newaxis] / np.sqrt(initial_disturbance_energy)  # shape (nx, ny, nz, n_traj)
    eta0 = eta0[:, :, :, np.newaxis] / np.sqrt(initial_disturbance_energy)  # shape (nx, ny, nz, n_traj)
    
    print("Initial disturbance kinetic energy normalized to 1: ", fom.inner_product_3D(v0[:, :, :, 0], eta0[:, :, :, 0], v0[:, :, :, 0], eta0[:, :, :, 0]))
    
    traj_template = np.load(params.fname_traj_template)
    traj_template_dx = np.load(params.fname_traj_template_dx)
    fom.load_template(traj_template, traj_template_dx)
    
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
            plt.pcolormesh(params.x, params.z, v_slice_ycheck.T, cmap='bwr')
            plt.colorbar()
            plt.xlabel(r"$x$")
            plt.ylabel(r"$z$")
            plt.xlim(np.min(params.x), np.max(params.x))
            plt.ylim(np.min(params.z), np.max(params.z))
            plt.title(f"Normal velocity v at t={t_check}, y={params.y_check}")
            plt.tight_layout()
            plt.show()
            
            plt.figure(figsize=(10, 6))
            # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
            cs = plt.contour(params.x, params.z, v_slice_ycheck.T, colors='black', linewidths=0.6)
            # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
            # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
            # plt.colorbar()
            plt.xlabel(r"$x$")
            plt.ylabel(r"$z$")
            plt.xlim(np.min(params.x), np.max(params.x))
            plt.ylim(np.min(params.z), np.max(params.z))
            plt.title(f"Contours of normal velocity v at t={t_check}, y={params.y_check}")
            plt.tight_layout()
            plt.show()
            
            plt.figure(figsize=(10,6))
            # plt.contourf(x, z, v_slice_ycheck.T, levels=levels, cmap='jet')
            plt.pcolormesh(params.x, params.z, v_fitted_slice_ycheck.T, cmap='bwr')
            plt.colorbar()
            plt.xlabel(r"$x$")
            plt.ylabel(r"$z$")
            plt.xlim(np.min(params.x), np.max(params.x))
            plt.ylim(np.min(params.z), np.max(params.z))
            plt.title(f"Fitted normal velocity v at t={t_check}, y={params.y_check}")
            plt.tight_layout()
            plt.show()
            
            plt.figure(figsize=(10, 6))
            # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
            cs = plt.contour(params.x, params.z, v_fitted_slice_ycheck.T, colors='black', linewidths=0.6)
            # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
            # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
            # plt.colorbar()
            plt.xlabel(r"$x$")
            plt.ylabel(r"$z$")
            plt.xlim(np.min(params.x), np.max(params.x))
            plt.ylim(np.min(params.z), np.max(params.z))
            plt.title(f"Contours of fitted normal velocity v at t={t_check}, y={params.y_check}")
            plt.tight_layout()
            plt.show()
            
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
    
if __name__ == "__main__":
    main()