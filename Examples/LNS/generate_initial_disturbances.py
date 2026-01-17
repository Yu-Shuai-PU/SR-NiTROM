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
import sys
sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

import configs
import fom_class_LNS

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
    
    X, Y, Z = params.X, params.Y, params.Z
    
    for idx_traj_training in range(params.n_traj_training // 2): # 0, 1, ..., 11
        angle = np.deg2rad(idx_traj_training * (360 / (params.n_traj_training // 2)))  # rotation angle in radians to introduce spanwise variation
        Xprime = np.cos(angle) * X + np.sin(angle) * Z
        Zprime = -np.sin(angle) * X + np.cos(angle) * Z
        v0 = (vhat_TS[np.newaxis, :, np.newaxis] * np.exp(1j * kx_TS * Xprime) * np.exp(-(Xprime)**2 - (Zprime/4)**2)).real
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
        # v = cos(theta) * dpsi/dx + sin(theta) * dpsi/dz
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
        
        v0   = v0 / np.sqrt(initial_disturbance_energy)  # shape (nx, ny, nz, n_traj)
        eta0 = eta0 / np.sqrt(initial_disturbance_energy)  # shape (nx, ny, nz, n_traj)
        
        np.save(params.fname_traj_init%idx_traj_training, np.concatenate((v0.ravel(), eta0.ravel())))
        
        idx_ycheck = np.argmin(np.abs(params.y - params.y_check))
        
        v0_ycheck = v0[:, idx_ycheck, :]
        eta0_ycheck = eta0[:, idx_ycheck, :]
        v_min = np.min(v0_ycheck)
        v_max = np.max(v0_ycheck)
        # v_spacing = 1e-6  # 等高线间距
        
        eta_min = np.min(eta0_ycheck)
        eta_max = np.max(eta0_ycheck)
        # eta_spacing = 1e-6  # 等高线间距

        # 构造等高线 levels
        # levels = np.arange(v_min - v_spacing, v_max + v_spacing, v_spacing)
        plt.figure(figsize=(10,6))
        # plt.contourf(x, z, v0_ycheck.T, levels=levels, cmap='jet')
        plt.pcolormesh(params.x, params.z, v0_ycheck.T, cmap='bwr')
        plt.colorbar()
        plt.xlabel(r"$x$")
        plt.ylabel(r"$z$")
        plt.xlim(np.min(params.x), np.max(params.x))
        plt.ylim(np.min(params.z), np.max(params.z))
        plt.title(f"Initial Normal velocity v at y={params.y_check}")
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
        cs = plt.contour(params.x, params.z, v0_ycheck.T, colors='black', linewidths=0.6)
        # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
        # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
        # plt.colorbar()
        plt.xlabel(r"$x$")
        plt.ylabel(r"$z$")
        plt.xlim(np.min(params.x), np.max(params.x))
        plt.ylim(np.min(params.z), np.max(params.z))
        plt.title(f"Contours of initial normal velocity v at y={params.y_check}")
        plt.tight_layout()
        plt.show()
        
    # endregion
    
    # region 3: Generate initial disturbances consisting of 2 pairs of counterrotating streamwise vortices centered around a pair of oblique waves (no kx = 0 or kz = 0 components)
    # To enrich the training dataset, here we introduce rotation to rotate our benchmark initial condition to create more variations
    
    # Define the rotating coordinates
    # x' = cos(theta) * x + sin(theta) * z
    # z' = -sin(theta) * x + cos(theta) * z
    
    # To generate such initial disturbances, we first formulate the streamfunction field psi(x, y, z) as follows:
    # psi(x, y, z) = (1 - y^2)^2 * (x'/2) * z' * exp(-(x'/2)^2 - (z'/2)^2)
    
    # Then, based on the continuity equation, we can formulate u and w as follows:
    # u = sin(theta) * dpsi/dy
    # v = dpsi/dz' = -sin(theta) * dpsi/dx + cos(theta) * dpsi/dz
    # w = -cos(theta) * dpsi/dy
    
    # Finally, we obtain the normal vorticity
    # eta(x, y, z) = du/dz - dw/dx
    
    for idx_traj_training in range(params.n_traj_training // 2): # 0, 1, ..., 11
        angle = np.deg2rad(idx_traj_training * (360 / (params.n_traj_training // 2)))  # rotation angle in radians to introduce spanwise variation
        Xprime = np.cos(angle) * X + np.sin(angle) * Z
        Zprime = -np.sin(angle) * X + np.cos(angle) * Z
        psi0 = (1 - Y**2)**2 * (Xprime/2) * Zprime * np.exp(-(Xprime/2)**2 - (Zprime/2)**2)
        u0 = np.sin(angle) * fom.diff_1_y(psi0)
        v0 = - np.sin(angle) * fom.diff_x(psi0, order = 1) + np.cos(angle) * fom.diff_z(psi0, order = 1)
        w0 = - np.cos(angle) * fom.diff_1_y(psi0)
        eta0 = fom.diff_z(u0, order = 1) - fom.diff_x(w0, order = 1)
        initial_disturbance_energy = fom.inner_product_3D(v0, eta0, v0, eta0)
        print("Initial disturbance energy before normalization: %.6e" % initial_disturbance_energy)
    
        # compute the initial disturbance energy for normalization
        
        v0   = v0 / np.sqrt(initial_disturbance_energy)  # shape (nx, ny, nz, n_traj)
        eta0 = eta0 / np.sqrt(initial_disturbance_energy)  # shape (nx, ny, nz, n_traj)
        
        np.save(params.fname_traj_init%(idx_traj_training + params.n_traj_training // 2), np.concatenate((v0.ravel(), eta0.ravel())))
        
        idx_ycheck = np.argmin(np.abs(params.y - params.y_check))
        
        v0_ycheck = v0[:, idx_ycheck, :]
        eta0_ycheck = eta0[:, idx_ycheck, :]
        v_min = np.min(v0_ycheck)
        v_max = np.max(v0_ycheck)
        # v_spacing = 1e-6  # 等高线间距
        
        eta_min = np.min(eta0_ycheck)
        eta_max = np.max(eta0_ycheck)
        # eta_spacing = 1e-6  # 等高线间距

        # 构造等高线 levels
        # levels = np.arange(v_min - v_spacing, v_max + v_spacing, v_spacing)
        plt.figure(figsize=(10,6))
        # plt.contourf(x, z, v0_ycheck.T, levels=levels, cmap='jet')
        plt.pcolormesh(params.x, params.z, v0_ycheck.T, cmap='bwr')
        plt.colorbar()
        plt.xlabel(r"$x$")
        plt.ylabel(r"$z$")
        plt.xlim(np.min(params.x), np.max(params.x))
        plt.ylim(np.min(params.z), np.max(params.z))
        plt.title(f"Initial Normal velocity v at y={params.y_check}")
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        # cs = plt.contour(x, z, v_slice_ycheck.T, levels=levels, colors='black', linewidths=0.6)
        cs = plt.contour(params.x, params.z, v0_ycheck.T, colors='black', linewidths=0.6)
        # plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")  # 可选：在曲线上标出数值
        # plt.pcolormesh(x, z, eta_slice_ycheck.T, cmap='bwr')
        # plt.colorbar()
        plt.xlabel(r"$x$")
        plt.ylabel(r"$z$")
        plt.xlim(np.min(params.x), np.max(params.x))
        plt.ylim(np.min(params.z), np.max(params.z))
        plt.title(f"Contours of initial normal velocity v at y={params.y_check}")
        plt.tight_layout()
        plt.show()
        
    # endregion
    
if __name__ == "__main__":
    main()