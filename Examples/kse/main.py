import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys
import os

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from pymanopt.tools.diagnostics import check_gradient
plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
plt.rc('text.latex',preamble=r'\usepackage{amsmath}')

sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

from my_pymanopt_classes import myAdaptiveLineSearcher
import classes
import nitrom_functions 
import opinf_functions as opinf_fun
import troop_functions
import fom_class_kse

cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'

#%% # Instantiate KSE class and KSE time-stepper class

L = 2 * np.pi
nx = 40
x = np.linspace(0, L, num=nx, endpoint=False)
nu = 4/87

fom = fom_class_kse.KSE(L, nu, nx)

dt = 1e-3
T = 130
time = dt * np.linspace(0, int(T/dt), int(T/dt) + 1, endpoint=True)
tstep_kse_fom = fom_class_kse.time_step_kse(fom, time)

nsave = 10
tsave = time[::nsave]

traj_path = "./trajectories/"
data_path = "./data/"
os.makedirs(traj_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

#%% # Generate and save trajectory
fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_rhs = traj_path + "rhs_%03d.npy"
fname_time = traj_path + "time.npy"

amps = np.array([[-1, 2, 3, -4]])
n_traj = len(amps)
uIC = np.zeros((nx, n_traj))
for k in range (n_traj):
    uIC[:,k] = amps[k,0] * np.sin(x) + amps[k,1] * np.cos(2 * x) + amps[k,2] * np.cos(3 * x) + amps[k,3] * np.sin(4 * x)

plt.plot(x,uIC)
plt.xlabel('$x$')
plt.ylabel('$u(x,0)$')
plt.title('KSE Initial Condition')
plt.tight_layout()
plt.show()
# for k in range (n_traj):
        
#     print("Running simulation %d/%d"%(k,n_traj))

#     Ukj, tsave = tstep_kse_fom.time_step(uIC[:,k],nsave)
#     dUdtkj = np.zeros_like(Ukj)
#     for j in range (Ukj.shape[-1]):
#         dUdtkj[:,j] = fom.evaluate_fom_rhs(0.0, Ukj[:,j], np.zeros(Ukj.shape[0]))

#     weight = np.mean(np.linalg.norm(dUdtkj,axis=0)**2)

#     np.save(fname_traj%k,Ukj)
#     np.save(fname_rhs%k,dUdtkj)
#     np.save(fname_weight%k,[weight])

# np.save(traj_path + "time.npy",tsave)

# --- MODIFIED PLOTTING SECTION ---
# Find the indices corresponding to the time interval from 120s to 130s
start_time = 120
end_time = 130

Ukj = np.load(fname_traj % 0)  # Load the trajectory data for the first simulation
tsave = np.load(fname_time)    # Load the time data

# Create a boolean mask for the desired time range
time_mask = (tsave >= start_time) & (tsave <= end_time)

# Apply the mask to get the relevant slices of data and time
tsave_slice = tsave[time_mask]
Ukj_slice = Ukj[:, time_mask]

# Plot the trajectory for the specified time interval
plt.figure(figsize=(10, 6))
if tsave_slice.size > 0 and Ukj_slice.size > 0:
    # plt.contourf(tsave_slice, x, Ukj_slice, 100, cmap='jet')
    plt.contourf(x, tsave_slice, Ukj_slice.T)
    plt.colorbar(label='u(x,t)')
else:
    print("No data available in the specified time range (120s - 130s).")
    # You might want to plot the full range as a fallback
    # plt.contourf(x, tsave, Ukj.T, 100, cmap='jet')
    # plt.colorbar(label='u(x,t)')


plt.xlabel('$x$')
plt.ylabel('Time (s)')
plt.title('KSE FOM Trajectory (120s - 130s)')
plt.tight_layout()
plt.show()
