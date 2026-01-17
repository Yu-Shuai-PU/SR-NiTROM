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
    
    # endregion
    
    # region 2: Generate initial disturbances that are centered around the least-stable 2D Tollmien-Schlichting (TS) wave at given Re
    # To enrich the training dataset, here we introduce rotation to rotate our benchmark initial condition to create more variations
    
    # Define the rotating coordinates
    # x' = cos(theta) * x + sin(theta) * z
    # z' = -sin(theta) * x + cos(theta) * z
    # Thus, we can first write down the expression of normal velocity
    # v(x, y, z) = Real(vhat_TS(y) * exp(1j * alpha_TS * x') * exp(-(x'/sigma_x)^2) * exp(-(z'/sigma_z)^2))
    # Then, based on the continuity equation, we can formulate u and w as follows:
    
    # u = -cos(theta) * dpsi/dy
    # v = dpsi/dx' = cos(theta) * dpsi/dx + sin(theta) * dpsi/dz
    # w = -sin(theta) * dpsi/dy
    
    # Finally, we obtain the normal vorticity
    # eta(x, y, z) = du/dz - dw/dx = d(-cos(theta) * dpsi/dy)/dz - d(-sin(theta) * dpsi/dy)/dx = sin(theta) * d^2 psi/dxdy - cos(theta) * d^2 psi/dydz
    
    # the process then goes as follows:
    # 1. figure out the optimal wavenumber and the normal velocity profile of the least-stable 2D TS wave at given wavenumber and Reynolds number (kx_TS, vhat_TS(y))
    # 2. formulate the initial normal velocity field v(x, y, z) based on the above expression
    # 3. solve the streamfunction field psi(x, y, z) from the normal velocity field v(x, y, z)
    # 4. figure out the initial u and w field from the continuity equation
    # 5. figure out the initial normal vorticity field eta(x, y, z) from u and w fields
    
    kx_TS, vhat_TS, maximal_growth_rate = generate_IC_around_2D_TS(params, fom)
    
    print("Least-stable 2D TS wave at Re = %d has streamwise wavenumber kx = %.4f, maximal growth rate = %.6f" % (params.Re, kx_TS, maximal_growth_rate))
    
    # plt.figure()
    # plt.plot(params.y, np.real(vhat_TS), label='Real part')
    # plt.plot(params.y, np.imag(vhat_TS), label='Imaginary part')
    # plt.title('Normal velocity profile of least-stable 2D TS wave at Re = %d' % params.Re)
    # plt.xlabel('y')
    # plt.ylabel('v_hat(y)')
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    X, Y, Z = params.X, params.Y, params.Z
    angle = np.deg2rad(90)  # rotation angle in radians to introduce spanwise variation
    Xprime = np.cos(angle) * X + np.sin(angle) * Z
    Zprime = -np.sin(angle) * X + np.cos(angle) * Z
    v0 = (vhat_TS[np.newaxis, :, np.newaxis] * np.exp(1j * kx_TS * Xprime) * np.exp(-(Xprime)**2 - (Zprime/8)**2)).real
    
    # after we figure out v0(x, y, z), we then compute psi0
    # v = cos(theta) * dpsi/dx + sin(theta) * dpsi/dz
    # that means v_breve(kx, y, kz) = 1j * (kx * cos(theta) + kz * sin(theta)) * psi_breve(kx, y, kz)
    # or psi_breve(kx, y, kz) = -1j * v_breve(kx, y, kz) / (kx * cos(theta) + kz * sin(theta))
    v0_breve = fom.FFT_2D(v0)
    psi0_breve = np.zeros_like(v0_breve, dtype=complex)
    for idx_x in range(params.nx):
        kx = fom.kx[idx_x]
        for idx_z in range(params.nz):
            kz = fom.kz[idx_z]
            denom = kx * np.cos(angle) + kz * np.sin(angle)
            print("Processing kx = %.4f, kz = %.4f, denom = %.4f" % (kx, kz, denom))
            if np.abs(denom) > 1e-6:
                psi0_breve[idx_x, :, idx_z] = -1j * v0_breve[idx_x, :, idx_z] / denom
            else:
                psi0_breve[idx_x, :, idx_z] = 0.0
                
    # after we figure out psi0, we then compute u0 and w0 from continuity equation
    # u = -cos(theta) * dpsi/dy
    # w = -sin(theta) * dpsi/dy
    psi0 = fom.IFFT_2D(psi0_breve)
    u0 = - np.cos(angle) * fom.diff_1_y(psi0)
    w0 = - np.sin(angle) * fom.diff_1_y(psi0)
    v0 = np.cos(angle) * fom.diff_x(psi0, order = 1) + np.sin(angle) * fom.diff_z(psi0, order = 1)
    
    # finally, we compute eta0
    eta0 = fom.diff_z(u0, order = 1) - fom.diff_x(w0, order = 1)
    initial_disturbance_energy = fom.inner_product_3D(v0, eta0, v0, eta0)
    print("Initial disturbance energy before normalization: %.6e" % initial_disturbance_energy)
    
    # compute the initial disturbance energy for normalization
    
    v0 = v0[:, :, :, np.newaxis] / np.sqrt(initial_disturbance_energy)  # shape (nx, ny, nz, n_traj)
    eta0 = eta0[:, :, :, np.newaxis] / np.sqrt(initial_disturbance_energy)  # shape (nx, ny, nz, n_traj)
    
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
        
    # endregion
    
if __name__ == "__main__":
    main()