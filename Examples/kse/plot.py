import numpy as np 
import scipy 
import matplotlib.pyplot as plt

import sys
import os

"""

Examplar plotting codes for the following figures:

1. The contourf plots of KSE solutions during the time interval [0, T] for FOM, SR-Galerkin ROM and SR-NiTROM ROM.
2. The comparison of shifting amounts over time for FOM, SR-Galerkin ROM and SR-NiTROM ROM.
3. The comparison of relative errors over time for SR-Galerkin ROM and SR-NiTROM ROM.
4. The comparison of the initial condition for FOM, POD-projected FOM state, and data-driven projection of the FOM state (as in the SR-NiTROM).
    
"""

def main():
    # Define paths
    path_to_plots = "./plots/"
    path_to_solutions = "./solutions/"
    os.makedirs(path_to_plots, exist_ok=True)

    # Example plotting code for relative errors over time
    T = 4
    n_ROM = 10
    n_x = 40
    L = 2 * np.pi
    x = np.linspace(0, L, n_x, endpoint=False)
    
    sol_idx = 1

    cmap_name = 'bwr'  # Colormap for contour plots
    contourf_vmin = -16
    contourf_vmax = 16
    contourf_levels = np.linspace(contourf_vmin, contourf_vmax, 9)
    fontsize = 20
    time_ticks = [0, int(T / 4), int(T / 2), int(3 * T / 4), int(T)]
    shift_amount_ticks = [-1, -0.5, 0, 0.5, 1]
    shift_speed_ticks = [-4, -2, 0, 2, 4]
    
    # region 1: plot the contourf plots of KSE solutions

    # Load time and the specified trajectory data
    time = np.load(path_to_solutions + "time.npy")
    sol_FOM = np.load(path_to_solutions + f"sol_{sol_idx:03d}.npy")
    sol_SRG = np.load(path_to_solutions + f"sol_SR_Galerkin_{sol_idx:03d}.npy")
    sol_SRN = np.load(path_to_solutions + f"sol_SR_NiTROM_{sol_idx:03d}.npy")
    
    # Plot the FOM solution
    plt.figure(figsize=(10,6))
    plt.contourf(x, time, sol_FOM.T, cmap=cmap_name, levels = contourf_levels)
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=fontsize)
    plt.title(f"FOM Solution No. {sol_idx}", fontsize=fontsize)
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
    plt.title(f"SR-Galerkin Solution No. {sol_idx}", fontsize=fontsize)
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
    plt.title(f"SR-NiTROM Solution No. {sol_idx}", fontsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.xlabel("x", fontsize=fontsize)
    plt.yticks(time_ticks, fontsize = fontsize)
    plt.ylabel("t", fontsize=fontsize)
    plt.show()
    
    # region 2: plot the shifting amounts of KSE solutions

    shift_amount_FOM = np.load(path_to_solutions + f"shift_amount_{sol_idx:03d}.npy")
    shift_amount_SRG = np.load(path_to_solutions + f"shift_amount_SR_Galerkin_{sol_idx:03d}.npy")
    shift_amount_SRN = np.load(path_to_solutions + f"shift_amount_SR_NiTROM_{sol_idx:03d}.npy")
    
    plt.figure(figsize=(10,6))
    plt.plot(time, shift_amount_FOM, label="FOM", color='k')
    plt.plot(time, shift_amount_SRG, label="SR-Galerkin", color='r')
    plt.plot(time, shift_amount_SRN, label="SR-NiTROM", color='b')
    plt.title(f"Shifting Amounts over Time, Sol No. {sol_idx}", fontsize=fontsize)
    plt.xticks(time_ticks, fontsize = fontsize)
    plt.xlabel("t", fontsize=fontsize)
    plt.yticks(shift_amount_ticks, fontsize = fontsize)
    plt.ylabel("c(t)", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.show()
    
    # region 3: plot the shifting speeds of KSE solutions

    shift_speed_FOM = np.load(path_to_solutions + f"shift_speed_{sol_idx:03d}.npy")
    shift_speed_SRG = np.load(path_to_solutions + f"shift_speed_SR_Galerkin_{sol_idx:03d}.npy")
    shift_speed_SRN = np.load(path_to_solutions + f"shift_speed_SR_NiTROM_{sol_idx:03d}.npy")
    
    plt.figure(figsize=(10,6))
    plt.plot(time, shift_speed_FOM, label="FOM", color='k')
    plt.plot(time, shift_speed_SRG, label="SR-Galerkin", color='r')
    plt.plot(time, shift_speed_SRN, label="SR-NiTROM", color='b')
    plt.title(f"Shifting Speeds over Time, Sol No. {sol_idx}", fontsize=fontsize)
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
    plt.title(f"Relative Errors over Time, Sol No. {sol_idx}", fontsize=fontsize)
    plt.xlabel("t", fontsize=fontsize)
    plt.xticks(time_ticks, fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.ylabel(r"$\epsilon(t)$", fontsize=fontsize)
    plt.ylim(0, 1)
    plt.legend(fontsize=fontsize)
    plt.show()
    
    # region 5: plot the initial conditions of FOM, POD-projected FOM state, and data-driven projection of the FOM state

    plt.figure(figsize=(10,6))
    plt.plot(x, sol_FOM[:,0], label="FOM IC", color='k')
    plt.plot(x, sol_SRG[:,0], label="POD-projected IC", color='r')
    plt.plot(x, sol_SRN[:,0], label="Data-driven projected IC", color='b')
    plt.title(f"Initial Conditions Comparison, Sol No. {sol_idx}", fontsize=fontsize)
    plt.xlabel("x", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel("u(x, 0)", fontsize=fontsize)
    plt.yticks(fontsize = fontsize)
    plt.ylim(contourf_vmin, contourf_vmax)
    plt.legend(fontsize=fontsize)
    plt.show()

    # endregion
    
if __name__ == "__main__":
    main()