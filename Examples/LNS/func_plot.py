import matplotlib.pyplot as plt
import numpy as np

def plot_ROM_vs_FOM(opt_obj, traj_idx, fig_path, relative_error, relative_error_fitted,
                    disturbance_kinetic_energy_FOM, disturbance_kinetic_energy_SRG,
                    shifting_amount_SRG, shifting_amount_FOM,
                    shifting_speed_SRG, shifting_speed_FOM,
                    traj_fitted_proj, sol_SRG,
                    traj_FOM, traj_fitted_FOM, traj_SRG, traj_fitted_SRG, num_modes_to_plot,
                    nx, ny, nz, dt, nsave, x, y, z, t_check_list, y_check):


    plt.figure(figsize=(10,6))
    plt.semilogy(opt_obj.time, relative_error, label="Relative error over time")
    plt.semilogy(opt_obj.time, relative_error_fitted, label="Relative error fitted over time")
    plt.xlabel("Time")
    plt.ylabel("Relative error")
    plt.title(f"Relative error between SR-Galerkin ROM and FOM along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_path + "relative_error_traj_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, disturbance_kinetic_energy_FOM, label="Disturbance kinetic energy FOM")
    plt.plot(opt_obj.time, disturbance_kinetic_energy_SRG, label="Disturbance kinetic energy SRG")
    plt.xlabel("Time")
    plt.ylabel("Disturbance kinetic energy")
    plt.title(f"Disturbance kinetic energy along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_path + "disturbance_kinetic_energy_traj_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, shifting_amount_SRG, label="Shifting amount SRG")
    plt.plot(opt_obj.time, shifting_amount_FOM, label="Shifting amount FOM")
    plt.xlabel("Time")
    plt.ylabel("Shifting amount c(t)")
    plt.title(f"Shifting amount c(t) between SR-Galerkin ROM and FOM along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_path + "shifting_amount_traj_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, shifting_speed_SRG, label="Shifting speed SRG")
    plt.plot(opt_obj.time, shifting_speed_FOM, label="Shifting speed FOM")
    plt.xlabel("Time")
    plt.ylabel("Shifting speed c_dot(t)")
    plt.title(f"Shifting speed c_dot(t) between SR-Galerkin ROM and FOM along trajectory {traj_idx}")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_path + "shifting_speed_traj_%03d.png"%traj_idx)
    plt.close()
    
    ### Plot the leading POD modes amplitudes
    
    for idx_mode in range (num_modes_to_plot):
        plt.figure(figsize=(10,6))
        plt.plot(opt_obj.time, sol_SRG[idx_mode,:], label=f"SRG Mode {idx_mode} amplitude over time")
        plt.plot(opt_obj.time, traj_fitted_proj[idx_mode,:], label=f"FOM Mode {idx_mode} amplitude over time")
        plt.xlabel("Time")
        plt.ylim([-0.1, 0.1])
        plt.ylabel(f"Mode {idx_mode} amplitude")
        plt.title(f"Mode {idx_mode} amplitude over time along trajectory {traj_idx}")
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(fig_path + "mode_%02d_amplitude_traj_%03d.png"%(idx_mode, traj_idx))
        plt.close()
    
    # np.save(fname_traj_FOM%traj_idx,X_FOM)
    # np.save(fname_traj_fitted_FOM%traj_idx,X_fitted_FOM)
    # np.save(fname_shift_amount_FOM%traj_idx,c_FOM)
    # np.save(fname_shift_speed_FOM%traj_idx,cdot_FOM)    
    # np.save(fname_traj_SRG%traj_idx,X_SRG)
    # np.save(fname_traj_fitted_SRG%traj_idx,X_fitted_SRG)
    # np.save(fname_shift_amount_SRG%traj_idx,c_SRG)
    # np.save(fname_shift_speed_SRG%traj_idx,cdot_SRG)
    # np.save(fname_relative_error_SRG%traj_idx,relative_error)
    
    ### Plotting, things to be done:
    ### 1. switch from contourf to pcolormesh
    ### 2. apply Fourier spectral interpolation to plot a 40-mode solution on a 256-point grid for better visualization
    
    traj_FOM_v = traj_FOM[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
    traj_FOM_eta = traj_FOM[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
    traj_fitted_FOM_v = traj_fitted_FOM[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
    traj_fitted_FOM_eta = traj_fitted_FOM[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
    traj_SRG_v = traj_SRG[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
    traj_SRG_eta = traj_SRG[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
    traj_fitted_SRG_v = traj_fitted_SRG[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
    traj_fitted_SRG_eta = traj_fitted_SRG[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
    
    v_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list)))
    eta_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list)))
    v_fitted_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list)))
    eta_fitted_slice_ycheck_all_z_centered_FOM = np.zeros((nx, len(t_check_list)))
    v_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list)))
    eta_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list)))
    v_fitted_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list)))
    eta_fitted_slice_ycheck_all_z_centered_SRG = np.zeros((nx, len(t_check_list)))

    for t_check in t_check_list:

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
        
        v_fitted_slice_ycheck_all_z_centered_FOM[:, t_check_list.index(t_check)] = v_fitted_slice_FOM_ycheck[:, nz//2]
        eta_fitted_slice_ycheck_all_z_centered_FOM[:, t_check_list.index(t_check)] = eta_fitted_slice_FOM_ycheck[:, nz//2]
        v_slice_ycheck_all_z_centered_FOM[:, t_check_list.index(t_check)] = v_slice_FOM_ycheck[:, nz//2]
        eta_slice_ycheck_all_z_centered_FOM[:, t_check_list.index(t_check)] = eta_slice_FOM_ycheck[:, nz//2]
        v_fitted_slice_ycheck_all_z_centered_SRG[:, t_check_list.index(t_check)] = v_fitted_slice_SRG_ycheck[:, nz//2]
        eta_fitted_slice_ycheck_all_z_centered_SRG[:, t_check_list.index(t_check)] = eta_fitted_slice_SRG_ycheck[:, nz//2]
        v_slice_ycheck_all_z_centered_SRG[:, t_check_list.index(t_check)] = v_slice_SRG_ycheck[:, nz//2]
        eta_slice_ycheck_all_z_centered_SRG[:, t_check_list.index(t_check)] = eta_slice_SRG_ycheck[:, nz//2]

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
        plt.title(f"Contours of normal velocity v at t={t_check}, y={y_check}")
        plt.tight_layout()
        # plt.show()
        plt.savefig(fig_path + "contours_v_t_%03d_traj_%03d.png"%(t_check,traj_idx))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.contour(x, z, v_fitted_slice_FOM_ycheck.T, levels=levels, colors='black', linewidths=0.6)
        plt.contour(x, z, v_fitted_slice_SRG_ycheck.T, levels=levels, colors='red', linewidths=0.6, linestyles='dashed')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$z$")
        plt.xlim(np.min(x), np.max(x))
        plt.ylim(np.min(z), np.max(z))
        plt.title(f"Contours of fitted normal velocity v at t={t_check}, y={y_check}")
        plt.tight_layout()
        # plt.show()
        plt.savefig(fig_path + "contours_fitted_v_t_%03d_traj_%03d.png"%(t_check,traj_idx))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, eta_fitted_slice_ycheck_all_z_centered_SRG[:, t_check_list.index(t_check)], label=f"SRG t={t_check_list[t_check_list.index(t_check)]}")
        plt.plot(x, eta_fitted_slice_ycheck_all_z_centered_FOM[:, t_check_list.index(t_check)], '--', label=f"FOM t={t_check_list[t_check_list.index(t_check)]}")
        plt.xlabel(r"$x$")
        plt.ylabel(r"Fitted normal vorticity at y={}".format(y_check))
        plt.title(f"Fitted normal vorticity at y={y_check} at t={t_check}")
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(fig_path + "fitted_normal_vorticity_y_%03d_t_%03d_traj_%03d.png"%(y_check,t_check,traj_idx))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, eta_slice_ycheck_all_z_centered_SRG[:, t_check_list.index(t_check)], label=f"SRG t={t_check_list[t_check_list.index(t_check)]}")
        plt.plot(x, eta_slice_ycheck_all_z_centered_FOM[:, t_check_list.index(t_check)], '--', label=f"FOM t={t_check_list[t_check_list.index(t_check)]}")
        plt.xlabel(r"$x$")
        plt.ylabel(r"Normal vorticity at y={}".format(y_check))
        plt.title(f"Normal vorticity at y={y_check} at t={t_check}")
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(fig_path + "normal_vorticity_y_%03d_t_%03d_traj_%03d.png"%(y_check,t_check,traj_idx))
        plt.close()
        
        
    # plt.figure(figsize=(10,6))
    # for i in range (len(t_check_list)):
    #     plt.plot(x, eta_fitted_slice_ycheck_all_z_centered_SRG[:, i], label=f"SRG t={t_check_list[i]}")
    #     plt.plot(x, eta_fitted_slice_ycheck_all_z_centered_FOM[:, i], '--', label=f"FOM t={t_check_list[i]}")
    # plt.xlabel(r"$x$")
    # plt.ylabel(r"Fitted normal vorticity at y={}".format(y_check))
    # plt.title("Fitted normal vorticity at y={} over time".format(y_check))
    # plt.legend()
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(fig_path + "fitted_normal_vorticity_y_over_time_%03d.png"%y_check)
    # plt.close()
    
    # plt.figure(figsize=(10,6))
    # for i in range (len(t_check_list)):
    #     plt.plot(x, eta_slice_ycheck_all_z_centered_SRG[:, i], label=f"SRG t={t_check_list[i]}")
    #     plt.plot(x, eta_slice_ycheck_all_z_centered_FOM[:, i], '--', label=f"FOM t={t_check_list[i]}")
    # plt.xlabel(r"$x$")
    # plt.ylabel(r"Normal vorticity at y={}".format(y_check))
    # plt.title("Normal vorticity at y={} over time".format(y_check))
    # plt.legend()
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(fig_path + "normal_vorticity_y_over_time_%03d.png"%y_check)
    # plt.close()