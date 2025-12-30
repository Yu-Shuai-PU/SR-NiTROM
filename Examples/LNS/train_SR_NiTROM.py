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
# plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
# plt.rc('text.latex',preamble=r'\usepackage{amsmath}')

sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

from my_pymanopt_classes import myAdaptiveLineSearcher
import classes
import nitrom_functions 
import opinf_functions as opinf_fun
import troop_functions
import fom_class_LNS
from func_plot import plot_ROM_vs_FOM

# cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
# lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'

#%% # Instantiate KSE class and KSE time-stepper class

traj_path = "./trajectories/"
data_path = "./data/"
fig_path = "./figures/"
os.makedirs(traj_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)

fname_traj_template = data_path + "traj_template.npy"
fname_traj_template_dx = data_path + "traj_template_dx.npy"
fname_traj_template_dxx = data_path + "traj_template_dxx.npy"
fname_traj_init = data_path + "traj_init_%03d.npy" # for initial condition of u
fname_traj_init_fitted = data_path + "traj_init_fitted_%03d.npy" # for initial condition of u fitted
fname_traj = traj_path + "traj_%03d.npy" # for u
fname_traj_fitted = traj_path + "traj_fitted_%03d.npy" # for u fitted
fname_weight_traj = traj_path + "weight_traj_%03d.npy"
fname_weight_shift_amount = traj_path + "weight_shift_amount_%03d.npy"
fname_weight_shift_speed = traj_path + "weight_shift_speed_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy" # for du/dt
fname_deriv_fitted = traj_path + "deriv_fitted_%03d.npy" # for du/dt fitted
fname_shift_amount = traj_path + "shift_amount_%03d.npy" # for shifting amount
fname_shift_speed = traj_path + "shift_speed_%03d.npy" # for shifting speed
fname_time = traj_path + "time.npy"
#%% # Generate and save trajectory
n_traj = 1

Lx = 48
Ly = 2 # from -1 to 1
Lz = 24

nx = 96
ny = 65 # ny includes the boundary points when using Chebyshev grid
nz = 96

x = np.linspace(0, Lx, num=nx, endpoint=False)
y = np.cos(np.pi * np.linspace(0, ny - 1, num=ny) / (ny - 1))  # Chebyshev grid in y direction, location from 1 to -1
z = np.linspace(0, Lz, num=nz, endpoint=False)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

Re = 3000
# Define the base flow
U_base = 1 - y**2
U_base_dy = -2 * y
U_base_dyy = -2 * np.ones_like(y)

T = 200
dt = 0.5
nsave = 2 # sample interval
time = dt * np.linspace(0, int(T/dt), int(T/dt) + 1, endpoint=True)
tsave = time[::nsave]

traj_template = np.load(fname_traj_template)
traj_template_dx = np.load(fname_traj_template_dx)
traj_template_dxx = np.load(fname_traj_template_dxx)
fom = fom_class_LNS.LNS(Lx, Ly, Lz, nx, ny, nz, y, Re, U_base, U_base_dy, U_base_dyy)
fom.load_template(traj_template, traj_template_dx)
time = dt * np.linspace(0, int(T/dt), int(T/dt) + 1, endpoint=True)
tstep_kse_fom = fom_class_LNS.time_step_LNS(fom, time)

n_traj = 1

# endregion

# region 2: Simulations
pool_inputs = (MPI.COMM_WORLD, n_traj)
pool_kwargs = {'fname_time':fname_time, 'fname_traj':fname_traj,'fname_traj_fitted':fname_traj_fitted,
               'fname_X_template':fname_traj_template, 'fname_X_template_dx':fname_traj_template_dx, 'fname_X_template_dxx':fname_traj_template_dxx,
               'fname_deriv':fname_deriv,'fname_deriv_fitted':fname_deriv_fitted,
               'fname_shift_amount':fname_shift_amount,'fname_shift_speed':fname_shift_speed}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)
pool.load_data()

T_final = pool.time[-1]

r = 40 # ROM dimension, should account for 99.5% energy
timespan_percentage_POD = 1 # percentage of the entire timespan used for POD
snapshot_start_time_idx_POD = 0
snapshot_end_time_idx_POD = 1 + int(timespan_percentage_POD * (pool.n_snapshots - 1))
max_iterations = 20
leggauss_deg = 5
nsave_rom = 11 # nsave_rom = 1 + int(dt_sample/dt) = 1 + sample_interval
which_trajs = np.arange(0, pool.my_n_traj, 1)
which_times = np.arange(snapshot_start_time_idx_POD,snapshot_end_time_idx_POD,1)

poly_comp = [1] # polynomial degree for the ROM dynamics

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

# region 1: SR-POD-Galerkin ROM

Phi_POD, cumulative_energy_proportion = opinf_fun.perform_POD(pool,opt_obj,r,fom)
Psi_POD = Phi_POD.copy() # Here, <Phi_POD, Phi_POD>_inner_product_3D = I, where inner_product_3D is the customized inner product defined in fom_class_LNS.py

# To convert our customized inner product to standard L2 inner product, <q, q>_E = <Rq, Rq>_L2, we need to weight the projection bases:

Psi_POD_weighted  = fom.apply_sqrt_inner_product_weight(Psi_POD) # R * Psi_POD
Phi_POD_weighted  = Psi_POD_weighted.copy() # R * Phi_POD
PhiF_POD_weighted = Phi_POD_weighted @ scipy.linalg.inv(Psi_POD_weighted.T @ Phi_POD_weighted) # PhiF_POD = Phi_POD (Psi_POD.T * W * Phi_POD)^{-1}, W = R^T * R

print(f"relative difference between PhiF_POD and Phi_POD: {np.linalg.norm(PhiF_POD_weighted - Phi_POD_weighted) / np.linalg.norm(Phi_POD_weighted):.4e}") # should be very small, yes!
### Test the SR-Galerkin ROM simulation accuracy

Tensors_POD = fom.assemble_petrov_galerkin_tensors(Psi_POD_weighted, PhiF_POD_weighted) # A, p, s, M
fname_Phi_POD = data_path + "Phi_POD.npy"
np.save(fname_Phi_POD,Phi_POD)
fname_Psi_POD = data_path + "Psi_POD.npy"
np.save(fname_Psi_POD,Psi_POD)
fname_Psi_POD_weighted = data_path + "Psi_POD_weighted.npy"
np.save(fname_Psi_POD_weighted,Psi_POD_weighted)
fname_PhiF_POD_weighted = data_path + "Phi_POD_weighted.npy"
np.save(fname_PhiF_POD_weighted,Phi_POD_weighted)
fname_Tensors_POD = data_path + "Tensors_POD_Galerkin.npz"
np.savez(fname_Tensors_POD, *Tensors_POD)
disturbance_kinetic_energy_FOM = np.zeros(opt_obj.n_snapshots)
disturbance_kinetic_energy_SRG = np.zeros(opt_obj.n_snapshots)
relative_error = np.zeros(opt_obj.n_snapshots)                                             
relative_error_fitted = np.zeros(opt_obj.n_snapshots)
relative_error_space_time_SRG = np.zeros(n_traj)

num_modes_to_plot = 10

for k in range(pool.my_n_traj):
    traj_idx = k + pool.disps[pool.rank]
    print("Preparing SR-Galerkin simulation %d/%d"%(traj_idx,pool.n_traj))
    traj_SRG_init = Psi_POD_weighted.T@fom.apply_sqrt_inner_product_weight(opt_obj.X_fitted[k,:,0].reshape(-1))
    shifting_amount_SRG_init = opt_obj.c[k,0]

    sol_SRG = solve_ivp(opt_obj.evaluate_rom_rhs,
                    [opt_obj.time[0],opt_obj.time[-1]],
                    np.hstack((traj_SRG_init, shifting_amount_SRG_init)),
                    'RK45',
                    t_eval=opt_obj.time,
                    args=(np.zeros(r),) + Tensors_POD).y
    
    traj_fitted_SRG = fom.apply_inv_sqrt_inner_product_weight(PhiF_POD_weighted@sol_SRG[:-1,:])
    traj_fitted_SRG_v = traj_fitted_SRG[0 : nx * ny * nz, :].reshape((nx, ny, nz, -1))
    traj_fitted_SRG_eta = traj_fitted_SRG[nx * ny * nz : , :].reshape((nx, ny, nz, -1))
    shifting_amount_SRG = sol_SRG[-1,:]
    traj_SRG = np.zeros_like(traj_fitted_SRG)
    shifting_speed_SRG = np.zeros_like(shifting_amount_SRG)

    for j in range (len(opt_obj.time)):
        traj_SRG_v_vec = fom.shift_x_input_3D(traj_fitted_SRG_v[:, :, :, j], shifting_amount_SRG[j])
        traj_SRG_eta_vec = fom.shift_x_input_3D(traj_fitted_SRG_eta[:, :, :, j], shifting_amount_SRG[j])
        traj_SRG[:,j] = np.concatenate((traj_SRG_v_vec.ravel(), traj_SRG_eta_vec.ravel()))
        shifting_speed_SRG[j] = opt_obj.compute_shift_speed(sol_SRG[:-1,j], Tensors_POD)
        diff_SRG_FOM_fitted = traj_fitted_SRG[:,j] - opt_obj.X_fitted[k,:,j]
        diff_SRG_FOM        = traj_SRG[:,j] - opt_obj.X[k,:,j]
        disturbance_kinetic_energy_FOM[j] = fom.inner_product_3D(opt_obj.X_fitted[k,0 : nx * ny * nz,j].reshape((nx, ny, nz)),
                                                               opt_obj.X_fitted[k,nx * ny * nz : ,j].reshape((nx, ny, nz)),
                                                               opt_obj.X_fitted[k,0 : nx * ny * nz,j].reshape((nx, ny, nz)),
                                                               opt_obj.X_fitted[k,nx * ny * nz : ,j].reshape((nx, ny, nz)))
        disturbance_kinetic_energy_SRG[j] = fom.inner_product_3D(traj_fitted_SRG[0 : nx * ny * nz,j].reshape((nx, ny, nz)),
                                                               traj_fitted_SRG[nx * ny * nz : ,j].reshape((nx, ny, nz)),
                                                               traj_fitted_SRG[0 : nx * ny * nz,j].reshape((nx, ny, nz)),
                                                               traj_fitted_SRG[nx * ny * nz : ,j].reshape((nx, ny, nz)))
        relative_error_fitted[j] = fom.inner_product_3D(diff_SRG_FOM_fitted[0 : nx * ny * nz].reshape((nx, ny, nz)),
                                                 diff_SRG_FOM_fitted[nx * ny * nz : ].reshape((nx, ny, nz)),
                                                 diff_SRG_FOM_fitted[0 : nx * ny * nz].reshape((nx, ny, nz)),
                                                 diff_SRG_FOM_fitted[nx * ny * nz : ].reshape((nx, ny, nz))) / disturbance_kinetic_energy_FOM[j]
        relative_error[j] = fom.inner_product_3D(diff_SRG_FOM[0 : nx * ny * nz].reshape((nx, ny, nz)),
                                                 diff_SRG_FOM[nx * ny * nz : ].reshape((nx, ny, nz)),
                                                 diff_SRG_FOM[0 : nx * ny * nz].reshape((nx, ny, nz)),
                                                 diff_SRG_FOM[nx * ny * nz : ].reshape((nx, ny, nz))) / disturbance_kinetic_energy_FOM[j]
        
    traj_FOM = opt_obj.X[k,:,:]
    shifting_amount_FOM = opt_obj.c[k,:]
    shifting_speed_FOM = opt_obj.cdot[k,:]
    traj_fitted_FOM = opt_obj.X_fitted[k,:,:]
    traj_fitted_proj = Psi_POD_weighted.T @ fom.apply_sqrt_inner_product_weight(traj_fitted_FOM)
    
    ### plotting
    plot_ROM_vs_FOM(opt_obj, traj_idx, fig_path, relative_error, relative_error_fitted,
                    disturbance_kinetic_energy_FOM, disturbance_kinetic_energy_SRG,
                    shifting_amount_SRG, shifting_amount_FOM,
                    shifting_speed_SRG, shifting_speed_FOM,
                    traj_fitted_proj, sol_SRG,
                    traj_FOM, traj_fitted_FOM, traj_SRG, traj_fitted_SRG, num_modes_to_plot,
                    nx, ny, nz, dt, nsave, x, y, z)
    
# endregion

# region 2: SR-NiTROM ROM

# Gr_Phi = manifolds.Grassmann(nx, r)
# Gr_Psi = manifolds.Grassmann(nx, r)
# # Gr_Psi = manifolds.Stiefel(nx, r)
# Euc_A  = manifolds.Euclidean(r, r)
# Euc_B  = manifolds.Euclidean(r, r, r)
# Euc_p  = manifolds.Euclidean(r)
# Euc_Q  = manifolds.Euclidean(r, r)

# M = manifolds.Product([Gr_Phi, Gr_Psi, Euc_A, Euc_B, Euc_p, Euc_Q])
# cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)

# # Choose between POD-Galerkin initialization and previous training results
# if initialization == "POD-Galerkin":
#     print("Loading POD-Galerkin results as initialization")
#     point = (Phi_POD, Psi_POD) + Tensors_POD[:-2]
# elif initialization == "Previous NiTROM":
#     print("Loading previous NiTROM results as initialization (for curriculum learning)")
#     Phi_NiTROM = np.load(data_path + "Phi_NiTROM.npy")
#     Psi_NiTROM = np.load(data_path + "Psi_NiTROM.npy")
#     npzfile = np.load(data_path + "Tensors_NiTROM.npz")
#     Tensors_NiTROM = (
#         npzfile['arr_0'],
#         npzfile['arr_1'],
#         npzfile['arr_2'],
#         npzfile['arr_3'],
#         npzfile['arr_4'],
#         npzfile['arr_5']
#     )
#     point = (Phi_NiTROM, Psi_NiTROM) + Tensors_NiTROM[:-2]


# if k0 == 0:
#     costvec_NiTROM = []
#     gradvec_NiTROM = []

# for k in range(k0, k0 + kouter):

#     if training_objects == "tensors_and_bases":

#         if np.mod(k, 2) == 0:
#             which_fix = 'fix_bases'
#             inner_iter = kinner_tensor
#             line_searcher = myAdaptiveLineSearcher(contraction_factor = contraction_factor_tensor, # how much to reduce the step size in each iteration
#                                             sufficient_decrease = sufficient_decrease_rate_tensor, # how much decrease is enough to accept the step
#                                             max_iterations = max_iterations,
#                                             initial_step_size = initial_step_size_tensor)
#         else:
#             which_fix = 'fix_tensors'
#             inner_iter = kinner_basis
#             line_searcher = myAdaptiveLineSearcher(contraction_factor = contraction_factor_basis, # how much to reduce the step size in each iteration
#                                             sufficient_decrease = sufficient_decrease_rate_basis, # how much decrease is enough to accept the step
#                                             max_iterations = max_iterations,
#                                             initial_step_size = initial_step_size_basis)
#     elif training_objects == "tensors":
#         # for the control group, always optimize the tensors
#         which_fix = 'fix_bases'
#         inner_iter = kinner_tensor
#         line_searcher = myAdaptiveLineSearcher(contraction_factor = contraction_factor_tensor, # how much to reduce the step size in each iteration
#                                         sufficient_decrease = sufficient_decrease_rate_tensor, # how much decrease is enough to accept the step
#                                         max_iterations = max_iterations,
#                                         initial_step_size = initial_step_size_tensor)
    
#     opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
#     opt_obj_kwargs = {
#     'sol_template_dx': pool.sol_template_dx,
#     'sol_template_dxx': pool.sol_template_dxx,
#     'spatial_derivative_method': fom.take_derivative,
#     'inner_product_method': fom.inner_product,
#     'outer_product_method': fom.outer_product,
#     'relative_weight': relative_weight,
#     'which_fix': which_fix}
#     opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
    
#     print("Optimizing (%d/%d) with which_fix = %s"%(k+1,kouter,opt_obj.which_fix))
    
#     cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
#     problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
#     optimizer = optimizers.ConjugateGradient(max_iterations=inner_iter,min_step_size=1e-20,max_time=3600,\
#                                               line_searcher=line_searcher,log_verbosity=1,verbosity=2)
#     result = optimizer.run(problem,initial_point=point)
#     point = result.point

#     Phi_NiTROM = point[0]
#     Psi_NiTROM = point[1]

#     ## Compute the difference between Phi and Psi using the method of principle angles

#     Q_Phi = np.linalg.qr(Phi_NiTROM)[0]
#     Q_Psi = np.linalg.qr(Psi_NiTROM)[0]
#     cos_thetas = np.linalg.svd(Q_Phi.T@Q_Psi, compute_uv=False)
#     print(cos_thetas)

#     itervec_NiTROM_k = result.log["iterations"]["iteration"]
#     costvec_NiTROM_k = result.log["iterations"]["cost"]
#     gradvec_NiTROM_k = result.log["iterations"]["gradient_norm"]

#     if k == 0:
#         costvec_NiTROM.extend(costvec_NiTROM_k)
#         gradvec_NiTROM.extend(gradvec_NiTROM_k)
#     else:
#         costvec_NiTROM.extend(costvec_NiTROM_k[1:])
#         gradvec_NiTROM.extend(gradvec_NiTROM_k[1:])
        
# opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
# opt_obj_kwargs = {
#     'sol_template_dx': pool.sol_template_dx,
#     'sol_template_dxx': pool.sol_template_dxx,
#     'spatial_derivative_method': fom.take_derivative,
#     'inner_product_method': fom.inner_product,
#     'outer_product_method': fom.outer_product,
#     'relative_weight': relative_weight,
#     'which_fix': 'fix_none'}
# opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
# cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
# problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
# # check_gradient(problem,x=point)

# Phi_NiTROM, Psi_NiTROM = point[0:2]
# Tensors_NiTROM_trainable = tuple(point[2:])
# PhiF_NiTROM = Phi_NiTROM @ scipy.linalg.inv(Psi_NiTROM.T@Phi_NiTROM)
# PhiF_NiTROM_dx = opt_obj.take_derivative(PhiF_NiTROM, order = 1)
# cdot_denom_linear = np.zeros(r)
# udx_linear = Psi_NiTROM.T @ PhiF_NiTROM_dx
# u0_dx = opt_obj.sol_template_dx

# for i in range(r):
#     cdot_denom_linear[i] = opt_obj.inner_product(u0_dx, PhiF_NiTROM_dx[:, i])

# Tensors_NiTROM = Tensors_NiTROM_trainable + (cdot_denom_linear, udx_linear)

# fname_Phi_NiTROM = data_path + "Phi_NiTROM.npy"
# np.save(fname_Phi_NiTROM,Phi_NiTROM)
# fname_Psi_NiTROM = data_path + "Psi_NiTROM.npy"
# np.save(fname_Psi_NiTROM,Psi_NiTROM)
# fname_Tensors_NiTROM = data_path + "Tensors_NiTROM.npz"
# np.savez(fname_Tensors_NiTROM, *Tensors_NiTROM)

# fname_sol_SR_NiTROM = sol_path + "sol_SR_NiTROM_%03d.npy" # for u
# fname_sol_fitted_SR_NiTROM = sol_path + "sol_fitted_SR_NiTROM_%03d.npy"
# fname_shift_amount_SR_NiTROM = sol_path + "shift_amount_SR_NiTROM_%03d.npy" # for shifting amount
# fname_shift_speed_SR_NiTROM = sol_path + "shift_speed_SR_NiTROM_%03d.npy"

# relative_error_all_sol = 0.0
# relative_error_fitted_all_sol = 0.0

# test_trial_consistency_percent = 1 - np.linalg.norm(Phi_NiTROM - Psi_NiTROM)/np.linalg.norm(Phi_NiTROM)
# print("Test-trial difference of POD bases: %.4e%%"%(test_trial_consistency_percent*100))

# for k in range(pool.my_n_sol):
#     sol_idx = k + pool.disps[pool.rank]
#     print("Preparing SR-NiTROM simulation %d/%d"%(sol_idx,n_sol))
#     z_IC_NiTROM = Psi_NiTROM.T@opt_obj.sol_fitted[k,:,0].reshape(-1)
#     shift_amount_IC = opt_obj.shift_amount[k,0]

#     output_SR_NiTROM = solve_ivp(opt_obj.evaluate_rom_rhs,
#                                         [opt_obj.time[0],opt_obj.time[-1]],
#                                         np.hstack((z_IC_NiTROM, shift_amount_IC)),
#                                         'RK45',
#                                         t_eval=opt_obj.time,
#                                         args=(np.zeros(r),) + Tensors_NiTROM).y

#     sol_fitted_SR_NiTROM = PhiF_NiTROM@output_SR_NiTROM[:-1,:]
#     shift_amount_SR_NiTROM = output_SR_NiTROM[-1,:]
#     sol_SR_NiTROM = np.zeros_like(sol_fitted_SR_NiTROM)
#     shift_speed_SR_NiTROM = np.zeros_like(shift_amount_SR_NiTROM)
#     for j in range (len(opt_obj.time)):
#         sol_SR_NiTROM[:,j] = fom.shift(sol_fitted_SR_NiTROM[:,j], shift_amount_SR_NiTROM[j])
#         shift_speed_SR_NiTROM[j] = opt_obj.compute_shift_speed(output_SR_NiTROM[:r,j], Tensors_NiTROM)

#     np.save(fname_sol_SR_NiTROM%sol_idx,sol_SR_NiTROM)
#     np.save(fname_sol_fitted_SR_NiTROM%sol_idx,sol_fitted_SR_NiTROM)
#     np.save(fname_shift_amount_SR_NiTROM%sol_idx,shift_amount_SR_NiTROM)
#     np.save(fname_shift_speed_SR_NiTROM%sol_idx,shift_speed_SR_NiTROM)
    
#     sol_FOM = opt_obj.sol[k,:,:]
#     sol_fitted_FOM = opt_obj.sol_fitted[k,:,:]  
#     shift_amount_FOM = opt_obj.shift_amount[k,:]
    
#     relative_error = np.linalg.norm(sol_FOM - sol_SR_NiTROM)/np.linalg.norm(sol_FOM)
#     relative_error_fitted = np.linalg.norm(sol_fitted_FOM - sol_fitted_SR_NiTROM)/np.linalg.norm(sol_fitted_FOM)
#     # print("Relative error of SR-NiTROM: %.4e"%(relative_error))
#     # print("Relative error of fitted SR-NiTROM: %.4e"%(relative_error_fitted))

#     plt.figure(figsize=(10,6))
#     plt.contourf(x,opt_obj.time,sol_FOM.T, levels = np.linspace(-16, 16, 9))
#     plt.colorbar()
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$t$")
#     plt.tight_layout()
#     if sol_idx == 0:
#         plt.title(f"FOM solution, initial condition = uIC")
#     elif 1 <= sol_idx <= (n_sol - 1) // 2:
#         plt.title(f"FOM solution, initial condition = uIC + {perturbation_amp} * cos({sol_idx} * x)")
#     else:
#         plt.title(f"FOM solution, initial condition = uIC + {perturbation_amp} * sin({sol_idx - (n_sol-1)//2} * x)")
#     plt.savefig(fig_path + "sol_FOM_%03d.png"%sol_idx)
#     plt.close()

#     plt.figure(figsize=(10,6))
#     plt.contourf(x,opt_obj.time,sol_SR_NiTROM.T, levels = np.linspace(-16, 16, 9))
#     plt.colorbar()
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$t$")
#     plt.tight_layout()
#     if sol_idx == 0:
#         plt.title(f"SRN solution, error: {relative_error:.4e}, initial condition = uIC")
#     elif 1 <= sol_idx <= (n_sol - 1) // 2:
#         plt.title(f"SRN solution, error: {relative_error:.4e}, initial condition = uIC + {perturbation_amp} * cos({sol_idx} * x)")
#     else:
#         plt.title(f"SRN solution, error: {relative_error:.4e}, initial condition = uIC + {perturbation_amp} * sin({sol_idx - (n_sol-1)//2} * x)")
#     plt.savefig(fig_path + "sol_SRN_%03d.png"%sol_idx)
#     plt.close()

#     plt.figure(figsize=(10,6))
#     plt.contourf(x,opt_obj.time,(sol_FOM - sol_SR_NiTROM).T, levels = np.linspace(-16, 16, 9))
#     plt.colorbar()
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$t$")
#     plt.tight_layout()
#     if sol_idx == 0:
#         plt.title(f"SRN-FOM diff, error: {relative_error:.4e}, initial condition = uIC")
#     elif 1 <= sol_idx <= (n_sol - 1) // 2:
#         plt.title(f"SRN-FOM diff, error: {relative_error:.4e}, initial condition = uIC + {perturbation_amp} * cos({sol_idx} * x)")
#     else:
#         plt.title(f"SRN-FOM diff, error: {relative_error:.4e}, initial condition = uIC + {perturbation_amp} * sin({sol_idx - (n_sol-1)//2} * x)")
#     plt.savefig(fig_path + "sol_SRN_FOM_diff_%03d.png"%sol_idx)
#     plt.close()
    
#     plt.figure(figsize=(10,6))
#     plt.contourf(x,opt_obj.time,sol_fitted_FOM.T, levels = np.linspace(-16, 16, 9))
#     plt.colorbar()
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$t$")
#     plt.tight_layout()
#     if sol_idx == 0:
#         plt.title(f"Fitted FOM solution, initial condition = uIC")
#     elif 1 <= sol_idx <= (n_sol - 1) // 2:
#         plt.title(f"Fitted FOM solution, initial condition = uIC + {perturbation_amp} * cos({sol_idx} * x)")
#     else:
#         plt.title(f"Fitted FOM solution, initial condition = uIC + {perturbation_amp} * sin({sol_idx - (n_sol-1)//2} * x)")
#     plt.savefig(fig_path + "sol_FOM_fitted_%03d.png"%sol_idx)
#     plt.close()
    
#     plt.figure(figsize=(10,6))
#     plt.contourf(x,opt_obj.time,sol_fitted_SR_NiTROM.T, levels = np.linspace(-16, 16, 9))
#     plt.colorbar()
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$t$")
#     plt.tight_layout()
#     if sol_idx == 0:
#         plt.title(f"Fitted SRN solution, error: {relative_error_fitted:.4e}, initial condition = uIC")
#     elif 1 <= sol_idx <= (n_sol - 1) // 2:
#         plt.title(f"Fitted SRN solution, error: {relative_error_fitted:.4e}, initial condition = uIC + {perturbation_amp} * cos({sol_idx} * x)")
#     else:
#         plt.title(f"Fitted SRN solution, error: {relative_error_fitted:.4e}, initial condition = uIC + {perturbation_amp} * sin({sol_idx - (n_sol-1)//2} * x)")
#     plt.savefig(fig_path + "sol_SRN_fitted_%03d.png"%sol_idx)
#     plt.close()
    
#     plt.figure(figsize=(10,6))
#     plt.contourf(x,opt_obj.time,(sol_fitted_FOM - sol_fitted_SR_NiTROM).T, levels = np.linspace(-16, 16, 9))
#     plt.colorbar()
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$t$")
#     plt.tight_layout()
#     if sol_idx == 0:
#         plt.title(f"Fitted SRN-FOM diff, error: {relative_error_fitted:.4e}, initial condition = uIC")
#     elif 1 <= sol_idx <= (n_sol - 1) // 2:
#         plt.title(f"Fitted SRN-FOM diff, error: {relative_error_fitted:.4e}, initial condition = uIC + {perturbation_amp} * cos({sol_idx} * x)")
#     else:
#         plt.title(f"Fitted SRN-FOM diff, error: {relative_error_fitted:.4e}, initial condition = uIC + {perturbation_amp} * sin({sol_idx - (n_sol-1)//2} * x)")
#     plt.savefig(fig_path + "sol_SRN_FOM_fitted_diff_%03d.png"%sol_idx)
#     plt.close()

#     plt.figure(figsize=(10,6))
#     plt.plot(opt_obj.time, shift_amount_FOM, color='k', linewidth=2, label='FOM')
#     plt.plot(opt_obj.time, shift_amount_SR_POD_Galerkin, color='r', linewidth=2, label='SR-Galerkin ROM')
#     plt.plot(opt_obj.time, shift_amount_SR_NiTROM, color='b', linewidth=2, label='SR-NiTROM ROM')
#     plt.xlabel(r"$t$")
#     plt.ylabel(r"Shift amount")
#     plt.ylim([-np.pi, np.pi])
#     plt.legend()
#     if sol_idx == 0:
#         plt.title(f"Shift amount, initial condition = uIC")
#     elif 1 <= sol_idx <= (n_sol - 1) // 2:
#         plt.title(f"Shift amount, initial condition = uIC + {perturbation_amp} * cos({sol_idx} * x)")
#     else:
#         plt.title(f"Shift amount, initial condition = uIC + {perturbation_amp} * sin({sol_idx - (n_sol-1)//2} * x)")
#     plt.savefig(fig_path + "shift_amount_SRN_FOM_%03d.png"%sol_idx)
#     plt.close()
    
#     plt.figure(figsize=(10,6))
#     plt.plot(opt_obj.time, shift_speed_FOM, color='k', linewidth=2, label='FOM')
#     plt.plot(opt_obj.time, shift_speed_SR_POD_Galerkin, color='r', linewidth=2, label='SR-Galerkin ROM')
#     plt.plot(opt_obj.time, shift_speed_SR_NiTROM, color='b', linewidth=2, label='SR-NiTROM ROM')
#     plt.xlabel(r"$t$")
#     plt.ylabel(r"Shift speed")
#     plt.ylim([-2, 2])
#     plt.legend()
#     if sol_idx == 0:
#         plt.title(f"Shift speed, initial condition = uIC")
#     elif 1 <= sol_idx <= (n_sol - 1) // 2:
#         plt.title(f"Shift speed, initial condition = uIC + {perturbation_amp} * cos({sol_idx} * x)")
#     else:
#         plt.title(f"Shift speed, initial condition = uIC + {perturbation_amp} * sin({sol_idx - (n_sol-1)//2} * x)")
#     plt.savefig(fig_path + "shift_speed_SRN_FOM_%03d.png"%sol_idx)
#     plt.close()

# relative_error_SRN_all_sol = np.mean(np.asarray(pool.comm.allgather(relative_error)))
# relative_error_SRN_fitted_all_sol = np.mean(np.asarray(pool.comm.allgather(relative_error_fitted)))
# if pool.rank == 0:
#     print("Mean relative error of SR-Galerkin for all solutions: %.4e"%(relative_error_SRG_all_sol))
#     print("Mean relative error of fitted SR-Galerkin for all solutions: %.4e"%(relative_error_SRG_fitted_all_sol))
#     print("Mean relative error of SR-NiTROM for all solutions: %.4e"%(relative_error_SRN_all_sol))
#     print("Mean relative error of fitted SR-NiTROM for all solutions: %.4e"%(relative_error_SRN_fitted_all_sol))

# ### plot the training error

# plt.figure(figsize=(8,6))
# # plt.semilogy(costvec_NiTROM,'-o',color=cOPT,label='SR-NiTROM')
# plt.semilogy(costvec_NiTROM,'-o',color='blue',label='SR-NiTROM')
# plt.xlabel('Iteration')
# plt.ylabel('Cost function')
# plt.title('Training error')
# plt.legend()
# plt.tight_layout()
# plt.savefig(fig_path + f"training_error_NiTROM_start_time_{snapshot_start_time_NiTROM_training}_end_time_{snapshot_end_time_NiTROM_training}.png")
# plt.close()

# ### Save the training information
# # Combine data and create a header for saving
# training_data = np.array([kouter, kinner_basis, kinner_tensor, r, relative_weight, initial_step_size_basis, initial_step_size_tensor, sufficient_decrease_rate_basis, sufficient_decrease_rate_tensor, contraction_factor_basis, contraction_factor_tensor, max_iterations])
# header_str = "kouter, kinner_basis, kinner_tensor, r, relative_weight, initial_step_size_basis, initial_step_size_tensor, sufficient_decrease_rate_basis, sufficient_decrease_rate_tensor, contraction_factor_basis, contraction_factor_tensor, max_iterations"

# # Define a unique filename for each simulation's training info
# fname_training_info = data_path + "training_hyperparameters_information_NiTROM.txt"
# # Save the training information
# # Using .reshape(1, -1) to save it as a single row
# np.savetxt(fname_training_info, training_data.reshape(1, -1), delimiter=',', header=header_str, comments='')
# print(f"Saved training information to {fname_training_info}")

# # endregion
