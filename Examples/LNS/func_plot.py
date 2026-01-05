import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

def spectral_resample(data, x_old, z_old, fom, zoom_factor=4):
    """
    适配你自带 shift 和归一化功能的 fom.FFT_2D/IFFT_2D。
    
    参数:
        data: (Nx, Nz) 原始数据
        x_old, z_old: 原始坐标
        fom: 求解器对象 (自动处理 shift 和 norm)
        zoom_factor: 加密倍数
    """
    Nx, Nz = data.shape
    Nx_new, Nz_new = int(Nx * zoom_factor), int(Nz * zoom_factor)
    
    # 1. 升维: (Nx, Nz) -> (Nx, 1, Nz)
    # 因为你的 fom.FFT_2D 强制要求 3D 输入且对 axis=(0, 2) 操作
    data_3d = data[:, np.newaxis, :]
    
    # 2. FFT
    # 你的函数已经做好了 fftshift，所以出来的数据，零频已经在中心了
    # 形状: (Nx, 1, Nz)
    f_hat_3d = fom.FFT_2D(data_3d)
    
    # 3. 补零 (Padding)
    # 创建全 0 的新谱空间 (Nx_new, 1, Nz_new)
    f_hat_padded = np.zeros((Nx_new, 1, Nz_new), dtype=complex)
    
    # 计算新旧网格的中心位置
    cx, cz = Nx // 2, Nz // 2
    cx_new, cz_new = Nx_new // 2, Nz_new // 2
    
    # 计算切片位置，把旧的谱直接贴到新谱的中心
    # 注意：这里不需要再做 fftshift，因为 f_hat_3d 已经是中心化的
    start_x = cx_new - cx
    end_x = start_x + Nx
    start_z = cz_new - cz
    end_z = start_z + Nz
    
    # 核心操作：中间嵌入
    f_hat_padded[start_x:end_x, 0, start_z:end_z] = f_hat_3d[:, 0, :]
    
    # 4. IFFT
    # 你的 fom.IFFT_2D 内部自带 ifftshift 和 * (Nx*Nz) 的操作
    # 所以直接传进去即可，幅度会自动修正，不需要额外乘 zoom_factor
    data_new_3d = fom.IFFT_2D(f_hat_padded)
    
    # 5. 降维并取实部
    data_new = data_new_3d[:, 0, :].real
    
    # 6. 生成新坐标
    x_new = np.linspace(x_old.min(), x_old.max(), Nx_new)
    z_new = np.linspace(z_old.min(), z_old.max(), Nz_new)
    
    return data_new, x_new, z_new

def plot_SRG_vs_FOM(opt_obj, traj_idx, fig_path, relative_error_SRG_idx, relative_error_fitted_SRG_idx,
                disturbance_kinetic_energy_FOM_idx, disturbance_kinetic_energy_SRG_idx,
                shifting_amount_FOM_idx, shifting_amount_SRG_idx,
                shifting_speed_FOM_idx, shifting_speed_SRG_idx,
                traj_fitted_FOM_proj_POD_idx, sol_SRG_idx,
                traj_FOM_idx, traj_SRG_idx, 
                traj_fitted_FOM_idx, traj_fitted_SRG_idx,
                num_modes_to_plot, nx, ny, nz, dt, nsave,
                x, y, z, t_check_list_POD, y_check):

    plt.figure(figsize=(10,6))
    plt.semilogy(opt_obj.time, relative_error_SRG_idx[:len(opt_obj.time)], 'r-', label="Relative error")
    plt.semilogy(opt_obj.time, relative_error_fitted_SRG_idx[:len(opt_obj.time)], 'r--', label="Relative error (fitted snapshots)")
    plt.xlabel("Time t")
    plt.ylabel("Relative Error")
    plt.title(f"Relative error of FOM and SR-Galerkin ROM along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path + "relative_error_traj_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, disturbance_kinetic_energy_FOM_idx[:len(opt_obj.time)], 'k-', label="FOM")
    plt.plot(opt_obj.time, disturbance_kinetic_energy_SRG_idx[:len(opt_obj.time)], 'r--', label="SR-Galerkin ROM")
    plt.xlabel("Time t")
    plt.ylabel("Kinetic Energy E(t)")
    plt.title(f"Disturbance kinetic energy of FOM and SR-Galerkin ROM along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path + "disturbance_kinetic_energy_traj_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, shifting_amount_FOM_idx[:len(opt_obj.time)], 'k-', label="FOM")
    plt.plot(opt_obj.time, shifting_amount_SRG_idx[:len(opt_obj.time)], 'r--', label="SR-Galerkin ROM")
    plt.xlabel("Time t")
    plt.ylabel("c(t)")
    plt.title(f"Shifting amount of FOM and SR-Galerkin ROM along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path + "shifting_amount_traj_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, shifting_speed_FOM_idx[:len(opt_obj.time)], 'k-', label="FOM")
    plt.plot(opt_obj.time, shifting_speed_SRG_idx[:len(opt_obj.time)], 'r--', label="SR-Galerkin ROM")
    plt.xlabel("Time t")
    plt.ylabel("dc/dt")
    plt.title(f"Shifting speed of FOM and SR-Galerkin ROM along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path + "shifting_speed_traj_%03d.png"%traj_idx)
    plt.close()
    
    integrated_FOM = cumulative_trapezoid(shifting_speed_FOM_idx[:len(opt_obj.time)], 
                                      opt_obj.time, 
                                      initial=0)
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, shifting_amount_FOM_idx[:len(opt_obj.time)], 'k-', label="FOM c(t)")
    plt.plot(opt_obj.time, integrated_FOM + shifting_amount_FOM_idx[0], 'k--', label="Integrated FOM dc/dt")
    plt.xlabel("Time t")
    plt.ylabel("Shifting Amount c(t)")
    plt.title(f"Verification of shifting amount via integrating shifting speed along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    ### Plot the leading POD modes amplitudes
    
    for idx_mode in range (num_modes_to_plot):
        plt.figure(figsize=(10,6))
        plt.plot(opt_obj.time, traj_fitted_FOM_proj_POD_idx[idx_mode,:len(opt_obj.time)], 'k-', label=f"FOM")
        plt.plot(opt_obj.time, sol_SRG_idx[idx_mode,:len(opt_obj.time)], 'r--', label=f"SR-Galerkin ROM")
        plt.xlabel("Time t")
        plt.ylim([-0.2, 0.2])
        plt.ylabel(f"Amplitude")
        plt.title(f"Mode {idx_mode} amplitude of FOM projection and SR-Galerkin ROM prediction along trajectory {traj_idx}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path + "mode_%02d_amplitude_traj_%03d.png"%(idx_mode, traj_idx))
        plt.close()
        
    ### Plotting, things to be done:
    ### 1. switch from contourf to pcolormesh
    ### 2. apply Fourier spectral interpolation to plot a 40-mode solution on a 256-point grid for better visualization
    
    traj_FOM_v = traj_FOM_idx[0 : nx * ny * nz, :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_FOM_eta = traj_FOM_idx[nx * ny * nz : , :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_fitted_FOM_v = traj_fitted_FOM_idx[0 : nx * ny * nz, :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_fitted_FOM_eta = traj_fitted_FOM_idx[nx * ny * nz : , :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_SRG_v = traj_SRG_idx[0 : nx * ny * nz, :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_SRG_eta = traj_SRG_idx[nx * ny * nz : , :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_fitted_SRG_v = traj_fitted_SRG_idx[0 : nx * ny * nz, :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_fitted_SRG_eta = traj_fitted_SRG_idx[nx * ny * nz : , :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    
    v_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list_POD)))
    eta_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list_POD)))
    v_fitted_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list_POD)))
    eta_fitted_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list_POD)))
    v_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list_POD)))
    eta_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list_POD)))
    v_fitted_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list_POD)))
    eta_fitted_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list_POD)))

    for t_check in t_check_list_POD:

        idx_sample = int(t_check / (dt * nsave))
        v_slice_FOM = traj_FOM_v[:, :, :, idx_sample]
        eta_slice_FOM = traj_FOM_eta[:, :, :, idx_sample]
        v_fitted_slice_FOM = traj_fitted_FOM_v[:, :, :, idx_sample]
        eta_fitted_slice_FOM = traj_fitted_FOM_eta[:, :, :, idx_sample]
        v_slice_SRG = traj_SRG_v[:, :, :, idx_sample]
        eta_slice_SRG = traj_SRG_eta[:, :, :, idx_sample]
        v_fitted_slice_SRG = traj_fitted_SRG_v[:, :, :, idx_sample]
        eta_fitted_slice_SRG = traj_fitted_SRG_eta[:, :, :, idx_sample]
        
        idx_y_check = np.argmin(np.abs(y - y_check))
        v_slice_FOM_ycheck = v_slice_FOM[:, idx_y_check, :]
        v_fitted_slice_FOM_ycheck = v_fitted_slice_FOM[:, idx_y_check, :]
        eta_slice_FOM_ycheck = eta_slice_FOM[:, idx_y_check, :]
        eta_fitted_slice_FOM_ycheck = eta_fitted_slice_FOM[:, idx_y_check, :]
        v_slice_SRG_ycheck = v_slice_SRG[:, idx_y_check, :]
        v_fitted_slice_SRG_ycheck = v_fitted_slice_SRG[:, idx_y_check, :]
        eta_slice_SRG_ycheck = eta_slice_SRG[:, idx_y_check, :]
        eta_fitted_slice_SRG_ycheck = eta_fitted_slice_SRG[:, idx_y_check, :]
        
        v_fitted_slice_ycheck_all_z_centered_FOM[:, t_check_list_POD.index(t_check)] = v_fitted_slice_FOM_ycheck[:, nz//2]
        eta_fitted_slice_ycheck_all_z_centered_FOM[:, t_check_list_POD.index(t_check)] = eta_fitted_slice_FOM_ycheck[:, nz//2]
        v_slice_ycheck_all_z_centered_FOM[:, t_check_list_POD.index(t_check)] = v_slice_FOM_ycheck[:, nz//2]
        eta_slice_ycheck_all_z_centered_FOM[:, t_check_list_POD.index(t_check)] = eta_slice_FOM_ycheck[:, nz//2]
        v_fitted_slice_ycheck_all_z_centered_SRG[:, t_check_list_POD.index(t_check)] = v_fitted_slice_SRG_ycheck[:, nz//2]
        eta_fitted_slice_ycheck_all_z_centered_SRG[:, t_check_list_POD.index(t_check)] = eta_fitted_slice_SRG_ycheck[:, nz//2]
        v_slice_ycheck_all_z_centered_SRG[:, t_check_list_POD.index(t_check)] = v_slice_SRG_ycheck[:, nz//2]
        eta_slice_ycheck_all_z_centered_SRG[:, t_check_list_POD.index(t_check)] = eta_slice_SRG_ycheck[:, nz//2]

        v_min = np.min(v_slice_FOM_ycheck)
        v_max = np.max(v_slice_FOM_ycheck)
        v_spacing = np.abs(v_max - v_min) / 20  # 等高线间距
        
        eta_min = np.min(eta_slice_FOM_ycheck)
        eta_max = np.max(eta_slice_FOM_ycheck)
        eta_spacing = np.abs(eta_max - eta_min) / 20  # 等高线间距

        # 构造等高线 levels
        levels = np.arange(v_min - v_spacing, v_max + v_spacing, v_spacing)
        
        plt.figure(figsize=(10, 6))
        plt.contour(x, z, v_slice_FOM_ycheck.T, levels=levels, colors='black', linewidths=0.6)
        plt.contour(x, z, v_slice_SRG_ycheck.T, levels=levels, colors='red', linewidths=0.6, linestyles='dashed')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$z$")
        plt.xlim(np.min(x), np.max(x))
        plt.ylim(np.min(z), np.max(z))
        plt.title(f"Normal velocity of FOM (black) and SR-Galerkin ROM (red) at t={t_check}, y={y_check}")
        plt.tight_layout()
        plt.savefig(fig_path + "v_t_%03d_y_%03d_traj_%03d.png"%(t_check,y_check,traj_idx))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.contour(x, z, v_fitted_slice_FOM_ycheck.T, levels=levels, colors='black', linewidths=0.6)
        plt.contour(x, z, v_fitted_slice_SRG_ycheck.T, levels=levels, colors='red', linewidths=0.6, linestyles='dashed')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$z$")
        plt.xlim(np.min(x), np.max(x))
        plt.ylim(np.min(z), np.max(z))
        plt.title(f"Fitted normal velocity of FOM (black) and SR-Galerkin ROM (red) at t={t_check}, y={y_check}")
        plt.tight_layout()
        plt.savefig(fig_path + "fitted_v_t_%03d_y_%03d_traj_%03d.png"%(t_check,y_check,traj_idx))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, eta_slice_ycheck_all_z_centered_FOM[:, t_check_list_POD.index(t_check)], 'k-', label=f"FOM t={t_check}")
        plt.plot(x, eta_slice_ycheck_all_z_centered_SRG[:, t_check_list_POD.index(t_check)], 'r--', label=f"SR-Galerkin ROM t={t_check}")
        plt.xlabel(r"$x$")
        plt.ylabel(f"Normal vorticity at (y,z)=({y_check}, {z[nz//2]})")
        plt.title(f"Normal vorticity at (y,z)=({y_check}, {z[nz//2]}) at t={t_check}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path + "eta_t_%03d_y_%03d_z_%03d_traj_%03d.png"%(t_check,y_check,z[nz//2],traj_idx))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, eta_fitted_slice_ycheck_all_z_centered_FOM[:, t_check_list_POD.index(t_check)], 'k-', label=f"FOM t={t_check}")
        plt.plot(x, eta_fitted_slice_ycheck_all_z_centered_SRG[:, t_check_list_POD.index(t_check)], 'r--', label=f"SR-Galerkin ROM t={t_check}")
        plt.xlabel(r"$x$")
        plt.ylabel(f"Fitted normal vorticity at (y,z)=({y_check}, {z[nz//2]})")
        plt.title(f"Fitted normal vorticity at (y,z)=({y_check}, {z[nz//2]}) at t={t_check}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path + "fitted_eta_t_%03d_y_%03d_z_%03d_traj_%03d.png"%(t_check,y_check,z[nz//2],traj_idx))
        plt.close()
        
def plot_SRN_vs_FOM(opt_obj, traj_idx, fig_path, relative_error_SRG_idx, relative_error_SRN_idx,
                    relative_error_fitted_SRG_idx, relative_error_fitted_SRN_idx,
                    disturbance_kinetic_energy_FOM_idx, disturbance_kinetic_energy_SRG_idx, disturbance_kinetic_energy_SRN_idx,
                    shifting_amount_FOM_idx, shifting_amount_SRG_idx, shifting_amount_SRN_idx,
                    shifting_speed_FOM_idx, shifting_speed_SRG_idx, shifting_speed_SRN_idx,
                    traj_fitted_FOM_proj_POD_idx, traj_fitted_FOM_proj_NiTROM_idx, sol_SRG_idx, sol_SRN_idx,
                    traj_FOM_idx, traj_SRG_idx, traj_SRN_idx, 
                    traj_fitted_FOM_idx, traj_fitted_SRG_idx, traj_fitted_SRN_idx,
                    num_modes_to_plot, nx, ny, nz, dt, nsave,
                    x, y, z, t_check_list_SRN, y_check):

    plt.figure(figsize=(10,6))
    plt.semilogy(opt_obj.time, relative_error_SRG_idx[:len(opt_obj.time)], 'r-', label="Relative error, SR-Galerkin ROM")
    plt.semilogy(opt_obj.time, relative_error_fitted_SRG_idx[:len(opt_obj.time)], 'r--', label="Relative error, SR-Galerkin ROM (fitted snapshots)")
    plt.semilogy(opt_obj.time, relative_error_SRN_idx[:len(opt_obj.time)], 'b-', label="Relative error, SR-NiTROM")
    plt.semilogy(opt_obj.time, relative_error_fitted_SRN_idx[:len(opt_obj.time)], 'b--', label="Relative error, SR-NiTROM (fitted snapshots)")
    plt.xlabel("Time t")
    plt.ylabel("Relative Error")
    plt.title(f"Relative error of FOM, SR-Galerkin ROM, and SR-NiTROM along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path + "relative_error_traj_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, disturbance_kinetic_energy_FOM_idx[:len(opt_obj.time)], 'k-', label="FOM")
    plt.plot(opt_obj.time, disturbance_kinetic_energy_SRG_idx[:len(opt_obj.time)], 'r--', label="SR-Galerkin ROM")
    plt.plot(opt_obj.time, disturbance_kinetic_energy_SRN_idx[:len(opt_obj.time)], 'b--', label="SR-NiTROM")
    plt.xlabel("Time t")
    plt.ylabel("Kinetic Energy E(t)")
    plt.title(f"Disturbance kinetic energy of FOM, SR-Galerkin ROM and SR-NiTROM along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path + "disturbance_kinetic_energy_traj_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, shifting_amount_FOM_idx[:len(opt_obj.time)], 'k-', label="FOM")
    plt.plot(opt_obj.time, shifting_amount_SRG_idx[:len(opt_obj.time)], 'r--', label="SR-Galerkin ROM")
    plt.plot(opt_obj.time, shifting_amount_SRN_idx[:len(opt_obj.time)], 'b--', label="SR-NiTROM")
    plt.xlabel("Time t")
    plt.ylabel("c(t)")
    plt.title(f"Shifting amount of FOM, SR-Galerkin ROM and SR-NiTROM along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path + "shifting_amount_traj_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, shifting_speed_FOM_idx[:len(opt_obj.time)], 'k-', label="FOM")
    plt.plot(opt_obj.time, shifting_speed_SRG_idx[:len(opt_obj.time)], 'r--', label="SR-Galerkin ROM")
    plt.plot(opt_obj.time, shifting_speed_SRN_idx[:len(opt_obj.time)], 'b--', label="SR-NiTROM")
    plt.xlabel("Time t")
    plt.ylabel("dc/dt")
    plt.title(f"Shifting speed of FOM, SR-Galerkin ROM and SR-NiTROM along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path + "shifting_speed_traj_%03d.png"%traj_idx)
    plt.close()
    
    ### Plot the leading POD modes amplitudes
    
    for idx_mode in range (num_modes_to_plot):
        plt.figure(figsize=(10,6))
        plt.plot(opt_obj.time, traj_fitted_FOM_proj_POD_idx[idx_mode,:len(opt_obj.time)], 'r-', label=f"POD projection of FOM")
        plt.plot(opt_obj.time, traj_fitted_FOM_proj_NiTROM_idx[idx_mode,:len(opt_obj.time)], 'b-', label=f"NiTROM oblique projection of FOM")
        plt.plot(opt_obj.time, sol_SRG_idx[idx_mode,:len(opt_obj.time)], 'r--', label=f"SR-Galerkin ROM prediction")
        plt.plot(opt_obj.time, sol_SRN_idx[idx_mode,:len(opt_obj.time)], 'b--', label=f"SR-NiTROM prediction")
        plt.xlabel("Time t")
        plt.ylim([-0.2, 0.2])
        plt.ylabel(f"Amplitude")
        plt.title(f"Mode {idx_mode} amplitude of FOM, SR-Galerkin ROM and SR-NiTROM along trajectory {traj_idx}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path + "mode_%02d_amplitude_traj_%03d.png"%(idx_mode, traj_idx))
        plt.close()
        
    ### Plotting, things to be done:
    ### 1. switch from contourf to pcolormesh
    ### 2. apply Fourier spectral interpolation to plot a 40-mode solution on a 256-point grid for better visualization
    
    traj_FOM_v = traj_FOM_idx[0 : nx * ny * nz, :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_FOM_eta = traj_FOM_idx[nx * ny * nz : , :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_fitted_FOM_v = traj_fitted_FOM_idx[0 : nx * ny * nz, :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_fitted_FOM_eta = traj_fitted_FOM_idx[nx * ny * nz : , :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_SRG_v = traj_SRG_idx[0 : nx * ny * nz, :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_SRG_eta = traj_SRG_idx[nx * ny * nz : , :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_fitted_SRG_v = traj_fitted_SRG_idx[0 : nx * ny * nz, :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_fitted_SRG_eta = traj_fitted_SRG_idx[nx * ny * nz : , :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_SRN_v = traj_SRN_idx[0 : nx * ny * nz, :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_SRN_eta = traj_SRN_idx[nx * ny * nz : , :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_fitted_SRN_v = traj_fitted_SRN_idx[0 : nx * ny * nz, :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    traj_fitted_SRN_eta = traj_fitted_SRN_idx[nx * ny * nz : , :len(opt_obj.time)].reshape((nx, ny, nz, -1))
    
    v_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list_SRN)))
    eta_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list_SRN)))
    v_fitted_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list_SRN)))
    eta_fitted_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list_SRN)))
    v_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list_SRN)))
    eta_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list_SRN)))
    v_fitted_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list_SRN)))
    eta_fitted_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list_SRN)))
    v_slice_ycheck_all_z_centered_SRN = np.zeros((nx, len(t_check_list_SRN)))
    eta_slice_ycheck_all_z_centered_SRN = np.zeros((nx, len(t_check_list_SRN)))
    v_fitted_slice_ycheck_all_z_centered_SRN = np.zeros((nx, len(t_check_list_SRN)))
    eta_fitted_slice_ycheck_all_z_centered_SRN = np.zeros((nx, len(t_check_list_SRN)))

    for t_check in t_check_list_SRN:

        idx_sample = int(t_check / (dt * nsave))
        v_slice_FOM = traj_FOM_v[:, :, :, idx_sample]
        eta_slice_FOM = traj_FOM_eta[:, :, :, idx_sample]
        v_fitted_slice_FOM = traj_fitted_FOM_v[:, :, :, idx_sample]
        eta_fitted_slice_FOM = traj_fitted_FOM_eta[:, :, :, idx_sample]
        v_slice_SRG = traj_SRG_v[:, :, :, idx_sample]
        eta_slice_SRG = traj_SRG_eta[:, :, :, idx_sample]
        v_fitted_slice_SRG = traj_fitted_SRG_v[:, :, :, idx_sample]
        eta_fitted_slice_SRG = traj_fitted_SRG_eta[:, :, :, idx_sample]
        v_slice_SRN = traj_SRN_v[:, :, :, idx_sample]
        eta_slice_SRN = traj_SRN_eta[:, :, :, idx_sample]
        v_fitted_slice_SRN = traj_fitted_SRN_v[:, :, :, idx_sample]
        eta_fitted_slice_SRN = traj_fitted_SRN_eta[:, :, :, idx_sample]
        
        idx_y_check = np.argmin(np.abs(y - y_check))
        v_slice_FOM_ycheck = v_slice_FOM[:, idx_y_check, :]
        v_fitted_slice_FOM_ycheck = v_fitted_slice_FOM[:, idx_y_check, :]
        eta_slice_FOM_ycheck = eta_slice_FOM[:, idx_y_check, :]
        eta_fitted_slice_FOM_ycheck = eta_fitted_slice_FOM[:, idx_y_check, :]
        v_slice_SRG_ycheck = v_slice_SRG[:, idx_y_check, :]
        v_fitted_slice_SRG_ycheck = v_fitted_slice_SRG[:, idx_y_check, :]
        eta_slice_SRG_ycheck = eta_slice_SRG[:, idx_y_check, :]
        eta_fitted_slice_SRG_ycheck = eta_fitted_slice_SRG[:, idx_y_check, :]
        v_slice_SRN_ycheck = v_slice_SRN[:, idx_y_check, :]
        v_fitted_slice_SRN_ycheck = v_fitted_slice_SRN[:, idx_y_check, :]
        eta_slice_SRN_ycheck = eta_slice_SRN[:, idx_y_check, :]
        eta_fitted_slice_SRN_ycheck = eta_fitted_slice_SRN[:, idx_y_check, :]
        
        v_fitted_slice_ycheck_all_z_centered_FOM[:, t_check_list_SRN.index(t_check)] = v_fitted_slice_FOM_ycheck[:, nz//2]
        eta_fitted_slice_ycheck_all_z_centered_FOM[:, t_check_list_SRN.index(t_check)] = eta_fitted_slice_FOM_ycheck[:, nz//2]
        v_slice_ycheck_all_z_centered_FOM[:, t_check_list_SRN.index(t_check)] = v_slice_FOM_ycheck[:, nz//2]
        eta_slice_ycheck_all_z_centered_FOM[:, t_check_list_SRN.index(t_check)] = eta_slice_FOM_ycheck[:, nz//2]
        v_fitted_slice_ycheck_all_z_centered_SRG[:, t_check_list_SRN.index(t_check)] = v_fitted_slice_SRG_ycheck[:, nz//2]
        eta_fitted_slice_ycheck_all_z_centered_SRG[:, t_check_list_SRN.index(t_check)] = eta_fitted_slice_SRG_ycheck[:, nz//2]
        v_slice_ycheck_all_z_centered_SRG[:, t_check_list_SRN.index(t_check)] = v_slice_SRG_ycheck[:, nz//2]
        eta_slice_ycheck_all_z_centered_SRG[:, t_check_list_SRN.index(t_check)] = eta_slice_SRG_ycheck[:, nz//2]
        v_fitted_slice_ycheck_all_z_centered_SRN[:, t_check_list_SRN.index(t_check)] = v_fitted_slice_SRN_ycheck[:, nz//2]
        eta_fitted_slice_ycheck_all_z_centered_SRN[:, t_check_list_SRN.index(t_check)] = eta_fitted_slice_SRN_ycheck[:, nz//2]
        v_slice_ycheck_all_z_centered_SRN[:, t_check_list_SRN.index(t_check)] = v_slice_SRN_ycheck[:, nz//2]
        eta_slice_ycheck_all_z_centered_SRN[:, t_check_list_SRN.index(t_check)] = eta_slice_SRN_ycheck[:, nz//2]
        
        v_min = np.min(v_slice_FOM_ycheck)
        v_max = np.max(v_slice_FOM_ycheck)
        v_spacing = np.abs(v_max - v_min) / 20  # 等高线间距
        
        eta_min = np.min(eta_slice_FOM_ycheck)
        eta_max = np.max(eta_slice_FOM_ycheck)
        eta_spacing = np.abs(eta_max - eta_min) / 20  # 等高线间距

        # 构造等高线 levels
        levels = np.arange(v_min - v_spacing, v_max + v_spacing, v_spacing)
        
        plt.figure(figsize=(10, 6))
        plt.contour(x, z, v_slice_FOM_ycheck.T, levels=levels, colors='black', linewidths=0.6)
        plt.contour(x, z, v_slice_SRG_ycheck.T, levels=levels, colors='red', linewidths=0.6, linestyles='dashed')
        plt.contour(x, z, v_slice_SRN_ycheck.T, levels=levels, colors='blue', linewidths=0.6, linestyles='dotted')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$z$")
        plt.xlim(np.min(x), np.max(x))
        plt.ylim(np.min(z), np.max(z))
        plt.title(f"Normal velocity of FOM (black), SR-Galerkin ROM (red), and SR-NiTROM (blue) at t={t_check}, y={y_check}")
        plt.tight_layout()
        plt.savefig(fig_path + "v_t_%03d_y_%03d_traj_%03d.png"%(t_check,y_check,traj_idx))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.contour(x, z, v_fitted_slice_FOM_ycheck.T, levels=levels, colors='black', linewidths=0.6)
        plt.contour(x, z, v_fitted_slice_SRG_ycheck.T, levels=levels, colors='red', linewidths=0.6, linestyles='dashed')
        plt.contour(x, z, v_fitted_slice_SRN_ycheck.T, levels=levels, colors='blue', linewidths=0.6, linestyles='dotted')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$z$")
        plt.xlim(np.min(x), np.max(x))
        plt.ylim(np.min(z), np.max(z))
        plt.title(f"Fitted normal velocity of FOM (black), SR-Galerkin ROM (red), and SR-NiTROM (blue) at t={t_check}, y={y_check}")
        plt.tight_layout()
        plt.savefig(fig_path + "fitted_v_t_%03d_y_%03d_traj_%03d.png"%(t_check,y_check,traj_idx))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, eta_slice_ycheck_all_z_centered_FOM[:, t_check_list_SRN.index(t_check)], 'k-', label=f"FOM t={t_check}")
        plt.plot(x, eta_slice_ycheck_all_z_centered_SRG[:, t_check_list_SRN.index(t_check)], 'r--', label=f"SR-Galerkin ROM t={t_check}")
        plt.plot(x, eta_slice_ycheck_all_z_centered_SRN[:, t_check_list_SRN.index(t_check)], 'b--', label=f"SR-NiTROM t={t_check}")
        plt.xlabel(r"$x$")
        plt.ylabel(f"Normal vorticity at (y,z)=({y_check}, {z[nz//2]})")
        plt.title(f"Normal vorticity at (y={y_check}, z={z[nz//2]}) at t={t_check}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path + "eta_t_%03d_y_%03d_z_%03d_traj_%03d.png"%(t_check,y_check,z[nz//2],traj_idx))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, eta_fitted_slice_ycheck_all_z_centered_FOM[:, t_check_list_SRN.index(t_check)], 'k-', label=f"FOM t={t_check}")
        plt.plot(x, eta_fitted_slice_ycheck_all_z_centered_SRG[:, t_check_list_SRN.index(t_check)], 'r--', label=f"SR-Galerkin ROM t={t_check}")
        plt.plot(x, eta_fitted_slice_ycheck_all_z_centered_SRN[:, t_check_list_SRN.index(t_check)], 'b--', label=f"SR-NiTROM t={t_check}")
        plt.xlabel(r"$x$")
        plt.ylabel(f"Fitted normal vorticity at (y,z)=({y_check}, {z[nz//2]})")
        plt.title(f"Fitted normal vorticity at (y,z)=({y_check}, {z[nz//2]}) at t={t_check}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path + "fitted_eta_t_%03d_y_%03d_z_%03d_traj_%03d.png"%(t_check,y_check,z[nz//2],traj_idx))
        plt.close()