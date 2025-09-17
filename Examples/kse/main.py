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
fname_sol_init = data_path + "sol_init_%03d.npy" # for initial condition of u
fname_sol_init_fitted = data_path + "sol_init_fitted_%03d.npy" # for initial condition of u fitted
fname_sol = sol_path + "sol_%03d.npy" # for u
fname_sol_fitted = sol_path + "sol_fitted_%03d.npy" # for u fitted
fname_weight = sol_path + "weight_%03d.npy"
fname_rhs = sol_path + "rhs_%03d.npy" # for du/dt
fname_rhs_fitted = sol_path + "rhs_fitted_%03d.npy" # for du/dt fitted
fname_shift_amount = sol_path + "shift_amount_%03d.npy" # for shifting amount
fname_shift_speed = sol_path + "shift_speed_%03d.npy" # for shifting speed
fname_time = sol_path + "time.npy"

n_sol = 1

pool_inputs = (MPI.COMM_WORLD, n_sol)
pool_kwargs = {'fname_time':fname_time, 'fname_sol':fname_sol,'fname_sol_fitted':fname_sol_fitted,
               'fname_sol_init':fname_sol_init, 'fname_sol_init_fitted':fname_sol_init_fitted,
               'fname_rhs':fname_rhs,'fname_rhs_fitted':fname_rhs_fitted,
               'fname_shift_amount':fname_shift_amount,'fname_shift_speed':fname_shift_speed,
               'fname_weights':fname_weight}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)
pool.load_data()

u_template = np.cos(x)
u_template_dx = -np.sin(x)
r = 4

Phi_POD, cumulative_energy_proportion = opinf_fun.perform_POD(pool,r)
Psi_POD = Phi_POD.copy()
PhiF_POD = Phi_POD@scipy.linalg.inv(Psi_POD.T@Phi_POD)

poly_comp = [1, 2] # polynomial degree for the ROM dynamics
Tensors_POD = fom.assemble_petrov_galerkin_tensors(Phi_POD, Psi_POD, u_template_dx) # A, B, p, Q, s, M

#%% Simulate SR-Galerkin ROM

which_trajs = np.arange(0,pool.n_sol,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 10

# region 1: SR-Galerkin ROM

"""
# filename = data_path + "SROpInf_ROM_w_reproj.npz"

# with np.load(filename) as data:
#     POD_basis = data['POD_basis']
#     SR_OpInf_linear = data['SR_OpInf_linear']
#     SR_OpInf_bilinear = data['SR_OpInf_bilinear']
#     SR_OpInf_cdot_numer_linear = data['SR_OpInf_cdot_numer_linear']
#     SR_OpInf_cdot_numer_bilinear = data['SR_OpInf_cdot_numer_bilinear']
#     SR_OpInf_cdot_denom_linear = data['SR_OpInf_cdot_denom']
#     SR_OpInf_udx_linear = data['SR_OpInf_udx_linear']
#     template = data['template']
"""

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
# opt_obj_kwargs = {'stab_promoting_pen':1e-2,'stab_promoting_tf':20,'stab_promoting_ic':(np.random.randn(r),)}

opt_obj = classes.optimization_objects(*opt_obj_inputs)

fname_sol_SR_POD_Galerkin = sol_path + "sol_SR_Galerkin_%03d.npy" # for u
fname_sol_fitted_SR_POD_Galerkin = sol_path + "sol_fitted_SR_Galerkin_%03d.npy"
fname_shift_amount_SR_POD_Galerkin = sol_path + "shift_amount_SR_Galerkin_%03d.npy" # for shifting amount
fname_shift_speed_SR_POD_Galerkin = sol_path + "shift_speed_SR_Galerkin_%03d.npy"

for k in range(pool.my_n_sol):
    sol_idx = k + pool.disps[pool.rank]
    print("Preparing SR-Galerkin simulation %d/%d"%(sol_idx,n_sol))
    sol_IC = Psi_POD.T@pool.sol_init_fitted[k,:].reshape(-1)
    shift_amount_IC = pool.shift_amount[k,0]

    output_SR_POD_Galerkin = solve_ivp(opt_obj.evaluate_rom_rhs,
                                          [0,time[-1]],
                                          np.hstack((sol_IC, shift_amount_IC)),
                                          'RK45',
                                          t_eval=tsave,
                                          args=(np.zeros(r),) + Tensors_POD).y
    
    sol_fitted_SR_POD_Galerkin = PhiF_POD@output_SR_POD_Galerkin[:r,:]
    shift_amount_SR_POD_Galerkin = output_SR_POD_Galerkin[-1,:]
    sol_SR_POD_Galerkin = np.zeros_like(sol_fitted_SR_POD_Galerkin)
    shift_speed_SR_POD_Galerkin = np.zeros_like(shift_amount_SR_POD_Galerkin)
    for j in range (len(tsave)):
        sol_SR_POD_Galerkin[:,j] = fom_class_kse.shift(sol_fitted_SR_POD_Galerkin[:,j], shift_amount_SR_POD_Galerkin[j], L)
        shift_speed_SR_POD_Galerkin[j] = opt_obj.compute_shift_speed(output_SR_POD_Galerkin[:r,j], Tensors_POD)

    np.save(fname_sol_SR_POD_Galerkin%sol_idx,sol_SR_POD_Galerkin)
    np.save(fname_sol_fitted_SR_POD_Galerkin%sol_idx,sol_fitted_SR_POD_Galerkin)
    np.save(fname_shift_amount_SR_POD_Galerkin%sol_idx,shift_amount_SR_POD_Galerkin)
    np.save(fname_shift_speed_SR_POD_Galerkin%sol_idx,shift_speed_SR_POD_Galerkin)

plt.figure(figsize=(10,6))
plt.contourf(x,tsave,sol_SR_POD_Galerkin.T)
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$t$")
plt.title(r"SR-POD-Galerkin ROM")
plt.tight_layout()
plt.show()

# endregion

# region 2: SR-NiTROM ROM

#%% First load the tensors from SR-OpInf w/ reproj ROM 

filename = data_path + "Tensors_SROpInf_w_reproj.npz"

with np.load(filename) as data:
    POD_basis = data['POD_basis']
    SR_OpInf_linear = data['SR_OpInf_linear']
    SR_OpInf_bilinear = data['SR_OpInf_bilinear']
    SR_OpInf_cdot_numer_linear = data['SR_OpInf_cdot_numer_linear']
    SR_OpInf_cdot_numer_bilinear = data['SR_OpInf_cdot_numer_bilinear']
    SR_OpInf_cdot_denom_linear = data['SR_OpInf_cdot_denom']
    SR_OpInf_udx_linear = data['SR_OpInf_udx_linear']
    template = data['template']

Gr_Phi = manifolds.Grassmann(nx, r)
Gr_Psi = manifolds.Grassmann(nx, r)
Euc_A  = manifolds.Euclidean(r, r)
Euc_B  = manifolds.Euclidean(r, r, r)
Euc_p  = manifolds.Euclidean(r)
Euc_Q  = manifolds.Euclidean(r, r)

M = manifolds.Product([Gr_Phi, Gr_Psi, Euc_A, Euc_B, Euc_p, Euc_Q])
cost, grad, hess = nitrom_functions.nitrom_cost_and_derivative_factory(M, opt_obj, pool, fom)

