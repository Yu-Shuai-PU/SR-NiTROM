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
T = 10
time = dt * np.linspace(0, int(T/dt), int(T/dt) + 1, endpoint=True)
tstep_kse_fom = fom_class_kse.time_step_kse(fom, time)

nsave = 10
tsave = time[::nsave]

sol_path = "./solutions/"
data_path = "./data/"
os.makedirs(sol_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

#%% # Generate and save trajectory
fname_sol = sol_path + "sol_%03d.npy" # for u
fname_sol_fitted = sol_path + "sol_fitted_%03d.npy" # for u fitted
fname_weight = sol_path + "weight_%03d.npy"
fname_rhs = sol_path + "rhs_%03d.npy" # for du/dt
fname_rhs_fitted = sol_path + "rhs_fitted_%03d.npy" # for du/dt fitted
fname_shift_amount = sol_path + "shift_amount_%03d.npy" # for shifting amount
fname_shift_speed = sol_path + "shift_speed_%03d.npy" # for shifting speed
fname_time = sol_path + "time.npy"

# amps = np.array([[-1, 2, 3, -4]])
# n_traj = len(amps)
# uIC = np.zeros((nx, n_traj))
# for k in range (n_traj):
#     uIC[:,k] = amps[k,0] * np.sin(x) + amps[k,1] * np.cos(2 * x) + amps[k,2] * np.cos(3 * x) + amps[k,3] * np.sin(4 * x)

n_sol = 1
u_IC = np.zeros((nx, n_sol))
u_IC = np.loadtxt(data_path + "initial_condition_time_80.txt") # load the initial condition to be the snapshot at t = 80 starting from the initial condition -sin(x) + 2cos(2x) + 3cos(3x) - 4sin(4x)
u_IC = u_IC.reshape((-1,1))

u_template = np.cos(x)
u_template_dx = -np.sin(x)

for k in range (n_sol):

    print("Running simulation %d/%d"%(k,n_sol))

    sol, tsave = tstep_kse_fom.time_step(u_IC[:,k],nsave)
    sol_fitted, shift_amount = fom_class_kse.template_fitting(sol, u_template, L, nx)
    rhs = np.zeros_like(sol)
    rhs_fitted = np.zeros_like(sol_fitted)
    shift_speed = np.zeros_like(shift_amount)
    for j in range (sol.shape[-1]):
        rhs[:,j] = fom.evaluate_fom_rhs(0.0, sol[:,j], np.zeros(sol.shape[0]))
        rhs_fitted[:, j] = fom_class_kse.shift(rhs[:,j], -shift_amount[j], L)
        sol_fitted_slice_dx = fom.take_derivative(sol_fitted[:,j], order = 1)
        shift_speed[j] = fom_class_kse.compute_shift_speed_FOM(rhs_fitted[:,j], sol_fitted_slice_dx, u_template_dx)
    weight = np.mean(np.linalg.norm(rhs,axis=0)**2)
    
    np.save(fname_sol%k,sol)
    np.save(fname_sol_fitted%k,sol_fitted)
    np.save(fname_rhs%k,rhs)
    np.save(fname_rhs_fitted%k,rhs_fitted)
    np.save(fname_shift_amount%k,shift_amount)
    np.save(fname_shift_speed%k,shift_speed)
    np.save(fname_weight%k,weight)

np.save(sol_path + "time.npy",tsave)

plt.figure(figsize=(10,6))
plt.contourf(x,tsave,sol.T)
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$t$")
plt.title(r"Trajectory")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.contourf(x,tsave,sol_fitted.T)
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$t$")
plt.title(r"Trajectory")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.contourf(x,tsave,rhs.T)
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$t$")
plt.title(r"Trajectory")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.contourf(x,tsave,rhs_fitted.T)
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$t$")
plt.title(r"Trajectory")
plt.tight_layout()
plt.show()

#%% Compute POD basis of the fitted solutions

r = 4

fnames_sol_fitted = [fname_sol_fitted%(k) for k in range (n_sol)]
U_fitted = [np.load(fnames_sol_fitted[k]) for k in range (n_sol)]
n_snapshots = U_fitted[0].shape[1]
sol_fitted = np.zeros((nx,n_sol*n_snapshots))
for k in range (n_sol): sol_fitted[:,k*n_snapshots:(k+1)*n_snapshots] = U_fitted[k]

SVD_basis, singular_values, _ = scipy.linalg.svd(sol_fitted,full_matrices=False)

Phi_POD = SVD_basis[:,:r]
Psi_POD = Phi_POD.copy()
PhiF_POD = Phi_POD@scipy.linalg.inv(Psi_POD.T@Phi_POD)
cumulative_energy_proportion = 100 * np.cumsum(singular_values[:r]**2) / np.sum(singular_values**2)

filename = data_path + "SROpInf_ROM_w_reproj.npz"

with np.load(filename) as data:
    POD_basis = data['POD_basis']
    SR_OpInf_linear = data['SR_OpInf_linear']
    SR_OpInf_bilinear = data['SR_OpInf_bilinear']
    SR_OpInf_cdot_numer_linear = data['SR_OpInf_cdot_numer_linear']
    SR_OpInf_cdot_numer_bilinear = data['SR_OpInf_cdot_numer_bilinear']
    SR_OpInf_cdot_denom_linear = data['SR_OpInf_cdot_denom']
    SR_OpInf_udx_linear = data['SR_OpInf_udx_linear']
    template = data['template']

poly_comp = [1, 2] # polynomial degree for the ROM dynamics
Tensors_POD = fom.assemble_petrov_galerkin_tensors(Phi_POD, Psi_POD, u_template_dx) # A, B, p, Q, s, M

pool_inputs = (MPI.COMM_WORLD, n_sol, fname_time)
pool_kwargs = {'fname_sol':fname_sol,'fname_sol_fitted':fname_sol_fitted,
               'fname_rhs':fname_rhs,'fname_rhs_fitted':fname_rhs_fitted,
               'fname_shift_amount':fname_shift_amount,'fname_shift_speed':fname_shift_speed,
               'fname_weights':fname_weight}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)

#%% Simulate SR-Galerkin ROM

which_trajs = np.arange(0,n_sol,1)
which_times = np.arange(0,n_snapshots,1)
leggauss_deg = 5
nsave_rom = 10

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
# opt_obj_kwargs = {'stab_promoting_pen':1e-2,'stab_promoting_tf':20,'stab_promoting_ic':(np.random.randn(r),)}

opt_obj = classes.optimization_objects(*opt_obj_inputs)

sol_SR_POD_Galerkin_init_state = Psi_POD.T @ opt_obj.sol_fitted[0,:,0]
sol_SR_POD_Galerkin_init_shift_amount = opt_obj.shift_amount[0,0]

output_SR_POD_Galerkin = solve_ivp(opt_obj.evaluate_rom_rhs,
                                          [0,time[-1]],
                                          np.hstack((sol_SR_POD_Galerkin_init_state, sol_SR_POD_Galerkin_init_shift_amount)),
                                          'RK45',
                                          t_eval=tsave,
                                          args=(np.zeros(r),) + Tensors_POD).y

sol_fitted_SR_POD_Galerkin = PhiF_POD@output_SR_POD_Galerkin[:r,:]
shift_amount_SR_POD_Galerkin = output_SR_POD_Galerkin[-1,:]
sol_SR_POD_Galerkin = np.zeros_like(sol_fitted_SR_POD_Galerkin)

for k in range (len(tsave)):
    sol_SR_POD_Galerkin[:,k] = fom_class_kse.shift(sol_fitted_SR_POD_Galerkin[:,k], shift_amount_SR_POD_Galerkin[k], L)

plt.figure(figsize=(10,6))
plt.contourf(x,tsave,sol_SR_POD_Galerkin.T)
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$t$")
plt.title(r"SR-Galerkin POD-ROM")
plt.tight_layout()
plt.show()

print("wait")
