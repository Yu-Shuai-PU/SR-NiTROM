import numpy as np 
import scipy 
import matplotlib.pyplot as plt

import sys
import os

"""

Todos:
1. Switch from contourf to pcolormesh for better performance if needed.
2. Use Fourier spectral interpolation (e.g: plot a 40-mode solution on a 256-mode grid for better visualization).

Examplar plotting codes for the following figures:

1. The contourf plots of KSE solutions during the time interval [0, T] for FOM, SR-Galerkin ROM and SR-NiTROM ROM.
2. The comparison of shifting amounts over time for FOM, SR-Galerkin ROM and SR-NiTROM ROM.
3. The comparison of relative errors over time for SR-Galerkin ROM and SR-NiTROM ROM.
4. The comparison of the initial condition for FOM, POD-projected FOM state, and data-driven projection of the FOM state (as in the SR-NiTROM).
    
"""

def clean_divergent_trajectory(sol_data, tolerance=1e-12, check_index=0):
    """
    Checks if a trajectory has stagnated (stopped changing between time steps), 
    indicating a solver failure, and replaces all subsequent data with np.inf.
    
    *** ASSUMES INPUT sol_data IS SHAPE (Nx, Nt) AND TRANSPOSES TO (Nt, Nx) ***

    Args:
        sol_data (np.ndarray): The trajectory data. 
                               Assumes shape is (Nx, Nt).
        tolerance (float): The threshold for determining "no change".
        check_index (int): The row index (spatial point) to check for stagnation 
                           after transposition, or the column index before transposition.
    
    Returns:
        tuple: (cleaned_sol_data, divergence_point_index)
    """
    
    # 1. Transpose the data: (Nx, Nt) -> (Nt, Nx)
    # This makes the time axis the first dimension (axis 0)
    sol_data_T = sol_data.T
    
    # 2. Extract the variable to check (Checking the time series of the check_index column)
    variable_to_check = sol_data_T[:, check_index]
    
    # 3. Calculate the difference between adjacent time steps (along the time axis)
    diffs = np.diff(variable_to_check)
    
    # 4. Find the indices where the difference is less than the tolerance
    stagnation_points = np.where(np.abs(diffs) < tolerance)[0]
    
    # Check if any stagnation point was found
    if len(stagnation_points) > 0:
        # Stagnation starts at the index i + 1 on the time axis (axis 0 of sol_data_T)
        stagnation_index_start = stagnation_points[0] + 1
        
        # 5. Fill from the stagnation index to the end with np.inf
        cleaned_sol_data_T = sol_data_T.copy()
        
        # Slicing the time axis (axis 0) and all spatial points (axis 1)
        cleaned_sol_data_T[stagnation_index_start:, :] = np.inf
        
        # 6. Re-transpose the data back to the original (Nx, Nt) shape before returning
        cleaned_sol_data = cleaned_sol_data_T.T
        
        print(f"Warning: Stagnation detected. Data from time step {stagnation_index_start} onwards set to np.inf.")
        return cleaned_sol_data, stagnation_index_start
    else:
        print("No significant stagnation detected.")
        return sol_data, -1

def main():
    # Define paths
    fig_path = "./figures_testing/"
    traj_path = "./trajectories_testing/"
    n_traj = 6
    time = np.load(traj_path + "time_testing.npy")
    os.makedirs(fig_path, exist_ok=True)

    # Example plotting code for relative errors over time
    # T = 4
    n_ROM = 10
    n_x = 40
    L = 2 * np.pi
    x = np.linspace(0, L, n_x, endpoint=False)
    
    traj_idx = 5

    cmap_name = 'bwr'  # Colormap for contour plots
    contourf_vmax = 16
    contourf_levels = np.linspace(-contourf_vmax, contourf_vmax, 9)
    fontsize = 20
    # time_ticks = [0, int(T / 4), int(T / 2), int(3 * T / 4), int(T)]
    shift_amount_ticks = [-1, -0.5, 0, 0.5, 1]
    shift_speed_ticks = [-4, -2, 0, 2, 4]
    time_ticks = [0, time[-1]/5, 2 * time[-1]/5, 3*time[-1]/5, 4*time[-1]/5, time[-1]]
    
    # region 1: plot the contourf plots of KSE solutions

    # Load the time and the data during the timespan of reconstructed training or testing trajectories
    
    T = time[-1]
    time_ticks = [0, T/5, 2*T/5, 3*T/5, 4*T/5, T]
    sol_FOM = np.load(traj_path + f"traj_FOM_{traj_idx:03d}.npy")
    sol_SRG = np.load(traj_path + f"traj_SRG_{traj_idx:03d}.npy")
    sol_SRN = np.load(traj_path + f"traj_SRN_{traj_idx:03d}.npy")
    
    sol_SRG, divergence_time_SRG = clean_divergent_trajectory(sol_SRG, tolerance=1e-12, check_index=0)
    sol_SRN, divergence_time_SRN = clean_divergent_trajectory(sol_SRN, tolerance=1e-12, check_index=0)
    
    # Plot the FOM solution
    plt.figure(figsize=(10,6))
    plt.contourf(x, time, sol_FOM.T, cmap=cmap_name, levels = contourf_levels)
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=fontsize)
    plt.title(f"FOM testing trajectory No. {traj_idx}", fontsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.xlabel("x", fontsize=fontsize)
    plt.yticks(time_ticks, fontsize = fontsize)
    plt.ylabel("t", fontsize=fontsize)
    plt.show()
    
    # Plot the SR-Galerkin solution (notice that when the shifting speed cdot blows up, we reset it to be 0 to finish simulation. Please be aware of this when you plot the solution)
    plt.figure(figsize=(10,6))
    plt.contourf(x, time, sol_SRG.T, cmap=cmap_name, levels = contourf_levels)
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=fontsize)
    plt.title(f"SR-Galerkin testing trajectory No. {traj_idx}", fontsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.xlabel("x", fontsize=fontsize)
    plt.yticks(time_ticks, fontsize = fontsize)
    plt.ylabel("t", fontsize=fontsize)
    plt.show()
    
    # Plot the SR-NiTROM solution
    plt.figure(figsize=(10,6))
    plt.contourf(x, time, sol_SRN.T, cmap=cmap_name, levels = contourf_levels)
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=fontsize)
    plt.title(f"SR-NiTROM testing trajectory No. {traj_idx}", fontsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.xlabel("x", fontsize=fontsize)
    plt.yticks(time_ticks, fontsize = fontsize)
    plt.ylabel("t", fontsize=fontsize)
    plt.show()
    
    # region 2: plot the shifting amounts of KSE solutions

    shift_amount_FOM = np.load(traj_path + f"shift_amount_FOM_{traj_idx:03d}.npy")
    shift_amount_SRG = np.load(traj_path + f"shift_amount_SRG_{traj_idx:03d}.npy")
    shift_amount_SRN = np.load(traj_path + f"shift_amount_SRN_{traj_idx:03d}.npy")
    
    shift_amount_SRG[divergence_time_SRG:] = np.inf
    shift_amount_SRN[divergence_time_SRN:] = np.inf
    
    plt.figure(figsize=(10,6))
    plt.plot(time, shift_amount_FOM, label="FOM", color='k')
    plt.plot(time, shift_amount_SRG, label="SR-Galerkin (diverged)", color='r')
    plt.plot(time, shift_amount_SRN, label="SR-NiTROM", color='b')
    plt.title(f"Shifting amounts over time, testing trajectory No. {traj_idx}", fontsize=fontsize)
    plt.xticks(time_ticks, fontsize = fontsize)
    plt.xlabel("t", fontsize=fontsize)
    plt.yticks(shift_amount_ticks, fontsize = fontsize)
    plt.ylabel("c(t)", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.show()
    
    # region 3: plot the shifting speeds of KSE solutions

    shift_speed_FOM = np.load(traj_path + f"shift_speed_FOM_{traj_idx:03d}.npy")
    shift_speed_SRG = np.load(traj_path + f"shift_speed_SRG_{traj_idx:03d}.npy")
    shift_speed_SRN = np.load(traj_path + f"shift_speed_SRN_{traj_idx:03d}.npy")
    
    shift_speed_SRG[divergence_time_SRG:] = np.inf
    shift_speed_SRN[divergence_time_SRN:] = np.inf
    
    plt.figure(figsize=(10,6))
    plt.plot(time, shift_speed_FOM, label="FOM", color='k')
    plt.plot(time, shift_speed_SRG, label="SR-Galerkin (diverged)", color='r')
    plt.plot(time, shift_speed_SRN, label="SR-NiTROM", color='b')
    plt.title(f"Shifting Speeds over Time, Sol No. {traj_idx}", fontsize=fontsize)
    plt.xlabel("t", fontsize=fontsize)
    plt.ylabel("dc(t)/dt", fontsize=fontsize)
    plt.xticks(time_ticks, fontsize = fontsize)
    plt.yticks(shift_speed_ticks, fontsize = fontsize)
    plt.ylim(min(shift_speed_ticks), max(shift_speed_ticks))
    plt.legend(fontsize=fontsize)
    plt.show()
    
    # region 4: plot the relative errors of KSE solutions

    relative_error_SRG = np.linalg.norm(sol_FOM - sol_SRG, axis=0) / np.linalg.norm(sol_FOM, axis=0)
    relative_error_SRN = np.linalg.norm(sol_FOM - sol_SRN, axis=0) / np.linalg.norm(sol_FOM, axis=0)
    
    plt.figure(figsize=(10,6))
    plt.plot(time, relative_error_SRG, label="SR-Galerkin", color='r')
    plt.plot(time, relative_error_SRN, label="SR-NiTROM", color='b')
    plt.title(f"Relative Errors over Time, Sol No. {traj_idx}", fontsize=fontsize)
    plt.xlabel("t", fontsize=fontsize)
    plt.xticks(time_ticks, fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.ylabel(r"$\epsilon(t)$", fontsize=fontsize)
    plt.ylim(0, 1)
    plt.legend(fontsize=fontsize)
    plt.show()
    
    # region 5: plot the relative errors of all training SRG and SRN solutions to KSE in one figure, comparing their performances
    
    plt.figure(figsize=(10,6))
    for traj_idx in range(n_traj):
        sol_FOM = np.load(traj_path + f"traj_FOM_{traj_idx:03d}.npy")
        sol_SRG = np.load(traj_path + f"traj_SRG_{traj_idx:03d}.npy")
        sol_SRN = np.load(traj_path + f"traj_SRN_{traj_idx:03d}.npy")
        
        sol_SRG, _ = clean_divergent_trajectory(sol_SRG, tolerance=1e-12, check_index=0)
        sol_SRN, _ = clean_divergent_trajectory(sol_SRN, tolerance=1e-12, check_index=0)
        relative_error_SRG = np.linalg.norm(sol_FOM - sol_SRG, axis=0) / np.linalg.norm(sol_FOM, axis=0)
        relative_error_SRN = np.linalg.norm(sol_FOM - sol_SRN, axis=0) / np.linalg.norm(sol_FOM, axis=0)
        if traj_idx == 0:
            plt.plot(time, relative_error_SRG, label=f"SR-Galerkin", color='r')
            plt.plot(time, relative_error_SRN, label=f"SR-NiTROM", color='b')
        else:
            plt.plot(time, relative_error_SRG, color='r')
            plt.plot(time, relative_error_SRN, color='b')
        plt.title(f"Relative errors of testing trajectories over time", fontsize=fontsize)
        plt.xlabel("t", fontsize=fontsize)
        plt.xticks(time_ticks, fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.ylabel(r"$e(t)$", fontsize=fontsize)
        plt.ylim(0, 1)
        plt.legend(fontsize=fontsize, loc = 'upper right')
        
    plt.show()
    
    
    # # region 5: plot the initial conditions of FOM, POD-projected FOM state, and data-driven projection of the FOM state

    # plt.figure(figsize=(10,6))
    # plt.plot(x, sol_FOM[:,0], label="FOM IC", color='k')
    # plt.plot(x, sol_SRG[:,0], label="POD-projected IC", color='r')
    # plt.plot(x, sol_SRN[:,0], label="Data-driven projected IC", color='b')
    # plt.title(f"Initial Conditions Comparison, Sol No. {sol_idx}", fontsize=fontsize)
    # plt.xlabel("x", fontsize=fontsize)
    # plt.xticks(fontsize=fontsize)
    # plt.ylabel("u(x, 0)", fontsize=fontsize)
    # plt.yticks(fontsize = fontsize)
    # plt.ylim(contourf_vmin, contourf_vmax)
    # plt.legend(fontsize=fontsize)
    # plt.show()

    # endregion
    
if __name__ == "__main__":
    main()