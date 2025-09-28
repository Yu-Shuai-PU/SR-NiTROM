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

sol_template = np.cos(x)
sol_template_dx = -np.sin(x)

fom = fom_class_kse.KSE(L, nu, nx, sol_template, sol_template_dx)

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
fname_weight_sol = sol_path + "weight_sol_%03d.npy"
fname_weight_shift_amount = sol_path + "weight_shift_amount_%03d.npy"
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
               'fname_weight_sol':fname_weight_sol,'fname_weight_shift_amount':fname_weight_shift_amount}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)
pool.load_data()

sol_template     = np.cos(x)
sol_template_dx  = -np.sin(x)
sol_template_dxx = -np.cos(x)

r = 4

# snapshot_start_time = 8 * (pool.n_snapshots - 1) // 9
snapshot_start_time = 0
snapshot_end_time = pool.n_snapshots

which_trajs = np.arange(0,pool.my_n_sol,1)
which_times = np.arange(snapshot_start_time,snapshot_end_time,1)
leggauss_deg = 5
nsave_rom = 11
poly_comp = [1, 2] # polynomial degree for the ROM dynamics
opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj_kwargs = {
    'sol_template_dx': sol_template_dx,
    'sol_template_dxx': sol_template_dxx,
    'spatial_derivative_method': fom.take_derivative,
    'inner_product_method': fom.inner_product,
    'outer_product_method': fom.outer_product,
    'relative_weight':1.0,
    'stab_promoting_pen':1e-2,
    'stab_promoting_tf':20,
    'stab_promoting_ic':(np.random.randn(r),)}
opt_obj = classes.optimization_objects(*opt_obj_inputs, **opt_obj_kwargs)


Phi_POD, cumulative_energy_proportion = opinf_fun.perform_POD(pool,opt_obj,r)
Psi_POD = Phi_POD.copy()
PhiF_POD = Phi_POD@scipy.linalg.inv(Psi_POD.T@Phi_POD)

Tensors_POD = fom.assemble_petrov_galerkin_tensors(Phi_POD, Psi_POD) # A, B, p, Q, s, M

# region 1: SR-Galerkin ROM

# filename = data_path + "Tensors_SROpInf_w_reproj.npz"

# with np.load(filename) as data:
#     POD_basis = data['POD_basis']
#     SR_OpInf_linear = data['SR_OpInf_linear']
#     SR_OpInf_bilinear = data['SR_OpInf_bilinear']
#     SR_OpInf_cdot_numer_linear = data['SR_OpInf_cdot_numer_linear']
#     SR_OpInf_cdot_numer_bilinear = data['SR_OpInf_cdot_numer_bilinear']
#     SR_OpInf_cdot_denom_linear = data['SR_OpInf_cdot_denom']
#     SR_OpInf_udx_linear = data['SR_OpInf_udx_linear']
#     template = data['template']
# which_trajs = np.arange(0,pool.my_n_sol,1)
# which_times = np.arange(0,pool.n_snapshots,1)
# leggauss_deg = 5
# nsave_rom = 11
# poly_comp = [1, 2] # polynomial degree for the ROM dynamics
# opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
# opt_obj_kwargs = {
#     'sol_template_dx': sol_template_dx,
#     'sol_template_dxx': sol_template_dxx,
#     'spatial_derivative_method': fom.take_derivative,
#     'inner_product_method': fom.inner_product,
#     'outer_product_method': fom.outer_product,
#     'relative_weight':1.0,
#     'stab_promoting_pen':1e-2,
#     'stab_promoting_tf':20,
#     'stab_promoting_ic':(np.random.randn(r),)}
# opt_obj = classes.optimization_objects(*opt_obj_inputs, **opt_obj_kwargs)

# fname_sol_SR_POD_Galerkin = sol_path + "sol_SR_Galerkin_%03d.npy" # for u
# fname_sol_fitted_SR_POD_Galerkin = sol_path + "sol_fitted_SR_Galerkin_%03d.npy"
# fname_shift_amount_SR_POD_Galerkin = sol_path + "shift_amount_SR_Galerkin_%03d.npy" # for shifting amount
# fname_shift_speed_SR_POD_Galerkin = sol_path + "shift_speed_SR_Galerkin_%03d.npy"

# for k in range(pool.my_n_sol):
#     sol_idx = k + pool.disps[pool.rank]
#     print("Preparing SR-Galerkin simulation %d/%d"%(sol_idx,n_sol))
#     z_IC = Psi_POD.T@opt_obj.sol_fitted[k,:,0].reshape(-1)
#     shift_amount_IC = opt_obj.shift_amount[k,0]

#     output_SR_POD_Galerkin = solve_ivp(opt_obj.evaluate_rom_rhs,
#                                           [0,opt_obj.time[-1]],
#                                           np.hstack((z_IC, shift_amount_IC)),
#                                           'RK45',
#                                           t_eval=opt_obj.time,
#                                           args=(np.zeros(r),) + Tensors_POD).y
    
#     sol_fitted_SR_POD_Galerkin = PhiF_POD@output_SR_POD_Galerkin[:-1,:]
#     shift_amount_SR_POD_Galerkin = output_SR_POD_Galerkin[-1,:]
#     sol_SR_POD_Galerkin = np.zeros_like(sol_fitted_SR_POD_Galerkin)
#     shift_speed_SR_POD_Galerkin = np.zeros_like(shift_amount_SR_POD_Galerkin)
#     for j in range (len(tsave)):
#         sol_SR_POD_Galerkin[:,j] = fom.shift(sol_fitted_SR_POD_Galerkin[:,j], shift_amount_SR_POD_Galerkin[j])
#         shift_speed_SR_POD_Galerkin[j] = opt_obj.compute_shift_speed(output_SR_POD_Galerkin[:r,j], Tensors_POD)

#     np.save(fname_sol_SR_POD_Galerkin%sol_idx,sol_SR_POD_Galerkin)
#     np.save(fname_sol_fitted_SR_POD_Galerkin%sol_idx,sol_fitted_SR_POD_Galerkin)
#     np.save(fname_shift_amount_SR_POD_Galerkin%sol_idx,shift_amount_SR_POD_Galerkin)
#     np.save(fname_shift_speed_SR_POD_Galerkin%sol_idx,shift_speed_SR_POD_Galerkin)
    
# fname_Phi_POD = data_path + "Phi_POD.npy"
# np.save(fname_Phi_POD,Phi_POD)
# fname_Psi_POD = data_path + "Psi_POD.npy"
# np.save(fname_Psi_POD,Psi_POD)
# fname_Tensors_POD = data_path + "Tensors_POD_Galerkin.npz"
# np.savez(fname_Tensors_POD, *Tensors_POD)
    
# # filename = data_path + "Tensors_SROpInf_w_reproj.npz"

# # plt.figure(figsize=(10,6))
# # plt.contourf(x,tsave,sol_SR_POD_Galerkin.T)
# # plt.colorbar()
# # plt.xlabel(r"$x$")
# # plt.ylabel(r"$t$")
# # plt.tight_layout()
# # plt.show()

# sol_FOM = np.load(fname_sol%0)
# # print('final shift amount of rom: %.4f'%(shift_amount_SR_POD_Galerkin[-1]))
# # plt.figure(figsize=(10,6))
# # plt.contourf(x,tsave,(sol_FOM - sol_SR_POD_Galerkin).T)
# # plt.colorbar()
# # plt.xlabel(r"$x$")
# # plt.ylabel(r"$t$")
# # plt.tight_layout()
# # plt.show()
# relative_error = np.linalg.norm(sol_FOM - sol_SR_POD_Galerkin)/np.linalg.norm(sol_FOM)
# print("Relative error of SR-Galerkin ROM: %.4e"%(relative_error))
# # sol_FOM_fitted = np.load(fname_sol_fitted%0)
# # plt.figure(figsize=(10,6))
# # plt.contourf(x,tsave,(sol_FOM_fitted - sol_fitted_SR_POD_Galerkin).T)
# # plt.colorbar()
# # plt.xlabel(r"$x$")
# # plt.ylabel(r"$t$")
# # plt.tight_layout()
# # plt.show()
# # relative_error_fitted = np.linalg.norm(sol_FOM_fitted - sol_fitted_SR_POD_Galerkin)/np.linalg.norm(sol_FOM_fitted)
# # print("Relative error of SR-Galerkin ROM (fitted): %.4e"%(relative_error_fitted))

# # # finally, we plot the FOM shift amount and the ROM shift amount
# shift_amount_FOM = np.load(fname_shift_amount%0)
# # plt.figure()
# # plt.plot(tsave, shift_amount_FOM, color='k', linewidth=2, label='FOM')
# # plt.plot(tsave, shift_amount_SR_POD_Galerkin, color=cPOD, linestyle=lPOD, linewidth=2, label='SR-Galerkin ROM')
# # plt.xlabel(r"$t$")
# # plt.ylabel(r"Shift amount")
# # plt.legend()
# # plt.tight_layout()
# # plt.show()
# # endregion

# # region 2: SR-NiTROM ROM

# #%% First load the tensors from SR-OpInf w/ reproj ROM 

# # filename = data_path + "Tensors_SROpInf_w_reproj.npz"

# # with np.load(filename) as data:
# #     POD_basis = data['POD_basis']
# #     SR_OpInf_linear = data['SR_OpInf_linear']
# #     SR_OpInf_bilinear = data['SR_OpInf_bilinear']
# #     SR_OpInf_cdot_numer_linear = data['SR_OpInf_cdot_numer_linear']
# #     SR_OpInf_cdot_numer_bilinear = data['SR_OpInf_cdot_numer_bilinear']
# #     SR_OpInf_cdot_denom_linear = data['SR_OpInf_cdot_denom']
# #     SR_OpInf_udx_linear = data['SR_OpInf_udx_linear']
# #     template = data['template']

# Gr_Phi = manifolds.Grassmann(nx, r)
# Gr_Psi = manifolds.Grassmann(nx, r)
# # Gr_Psi = manifolds.Stiefel(nx, r)
# Euc_A  = manifolds.Euclidean(r, r)
# Euc_B  = manifolds.Euclidean(r, r, r)
# Euc_p  = manifolds.Euclidean(r)
# Euc_Q  = manifolds.Euclidean(r, r)

# M = manifolds.Product([Gr_Phi, Gr_Psi, Euc_A, Euc_B, Euc_p, Euc_Q])
# cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)

# line_searcher = myAdaptiveLineSearcher(contraction_factor = 0.4,
#                                         sufficient_decrease = 0.1,
#                                         max_iterations = 25,
#                                         initial_step_size = 3e-3)

# point = (Phi_POD, Psi_POD) + Tensors_POD[:-2]

# k0 = 0
# kouter = 4

# if k0 == 0:
#     costvec_NiTROM = []
#     gradvec_NiTROM = []
    
# for k in range(k0, k0 + kouter):
    
#     if np.mod(k, 2) == 0:    which_fix = 'fix_bases'
#     else:                   which_fix = 'fix_tensors'
    
#     opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
#     opt_obj_kwargs = {
#     'sol_template_dx': sol_template_dx,
#     'sol_template_dxx': sol_template_dxx,
#     'spatial_derivative_method': fom.take_derivative,
#     'inner_product_method': fom.inner_product,
#     'outer_product_method': fom.outer_product,
#     'relative_weight':1.0,
#     'which_fix': which_fix}
#     opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
    
#     print("Optimizing (%d/%d) with which_fix = %s"%(k+1,kouter,opt_obj.which_fix))
    
#     cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
#     problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
#     optimizer = optimizers.ConjugateGradient(max_iterations=4,min_step_size=1e-20,max_time=3600,\
#                                               line_searcher=line_searcher,log_verbosity=1,verbosity=2)
#     result = optimizer.run(problem,initial_point=point)
#     point = result.point

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
#     'sol_template_dx': sol_template_dx,
#     'sol_template_dxx': sol_template_dxx,
#     'spatial_derivative_method': fom.take_derivative,
#     'inner_product_method': fom.inner_product,
#     'outer_product_method': fom.outer_product,
#     'relative_weight':1.0,
#     'which_fix': 'fix_none'}
# opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
# cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
# problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
# check_gradient(problem,x=point)

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

# for k in range(pool.my_n_sol):
#     sol_idx = k + pool.disps[pool.rank]
#     print("Preparing SR-NiTROM simulation %d/%d"%(sol_idx,n_sol))
#     z_IC_NiTROM = Psi_NiTROM.T@opt_obj.sol_fitted[k,:,0].reshape(-1)
#     shift_amount_IC_NiTROM = opt_obj.shift_amount[k,0]

#     output_SR_NiTROM = solve_ivp(opt_obj.evaluate_rom_rhs,
#                                         [0,opt_obj.time[-1]],
#                                         np.hstack((z_IC_NiTROM, shift_amount_IC_NiTROM)),
#                                         'RK45',
#                                         t_eval=opt_obj.time,
#                                         args=(np.zeros(r),) + Tensors_NiTROM).y

#     sol_fitted_SR_NiTROM = PhiF_NiTROM@output_SR_NiTROM[:-1,:]
#     shift_amount_SR_NiTROM = output_SR_NiTROM[-1,:]
#     sol_SR_NiTROM = np.zeros_like(sol_fitted_SR_NiTROM)
#     shift_speed_SR_NiTROM = np.zeros_like(shift_amount_SR_NiTROM)
#     for j in range (len(tsave)):
#         sol_SR_NiTROM[:,j] = fom.shift(sol_fitted_SR_NiTROM[:,j], shift_amount_SR_NiTROM[j])
#         shift_speed_SR_NiTROM[j] = opt_obj.compute_shift_speed(output_SR_NiTROM[:r,j], Tensors_NiTROM)

#     np.save(fname_sol_SR_NiTROM%sol_idx,sol_SR_NiTROM)
#     np.save(fname_sol_fitted_SR_NiTROM%sol_idx,sol_fitted_SR_NiTROM)
#     np.save(fname_shift_amount_SR_NiTROM%sol_idx,shift_amount_SR_NiTROM)
#     np.save(fname_shift_speed_SR_NiTROM%sol_idx,shift_speed_SR_NiTROM)

# plt.figure(figsize=(10,6))
# plt.contourf(x,tsave,sol_SR_NiTROM.T)
# plt.colorbar()
# plt.xlabel(r"$x$")
# plt.ylabel(r"$t$")
# plt.tight_layout()
# plt.show()

# sol_FOM = np.load(fname_sol%0)
# print('final shift amount of rom: %.4f'%(shift_amount_SR_POD_Galerkin[-1]))
# plt.figure(figsize=(10,6))
# plt.contourf(x,tsave,(sol_FOM - sol_SR_POD_Galerkin).T)
# plt.colorbar()
# plt.xlabel(r"$x$")
# plt.ylabel(r"$t$")
# plt.tight_layout()
# plt.show()
# relative_error = np.linalg.norm(sol_FOM - sol_SR_NiTROM)/np.linalg.norm(sol_FOM)
# print("Relative error of SR-NiTROM: %.4e"%(relative_error))
# sol_FOM_fitted = np.load(fname_sol_fitted%0)
# plt.figure(figsize=(10,6))
# plt.contourf(x,tsave,(sol_FOM_fitted - sol_fitted_SR_NiTROM).T)
# plt.colorbar()
# plt.xlabel(r"$x$")
# plt.ylabel(r"$t$")
# plt.tight_layout()
# plt.show()
# relative_error_fitted = np.linalg.norm(sol_FOM_fitted - sol_fitted_SR_NiTROM)/np.linalg.norm(sol_FOM_fitted)
# print("Relative error of SR-NiTROM ROM (fitted): %.4e"%(relative_error_fitted))

# # # finally, we plot the FOM shift amount and the ROM shift amount
# shift_amount_FOM = np.load(fname_shift_amount%0)
# plt.figure()
# plt.plot(tsave, shift_amount_FOM, color='k', linewidth=2, label='FOM')
# plt.plot(tsave, shift_amount_SR_NiTROM, color=cPOD, linestyle=lPOD, linewidth=2, label='SR-NiTROM ROM')
# plt.xlabel(r"$t$")
# plt.ylabel(r"Shift amount")
# plt.legend()
# plt.tight_layout()
# plt.show()

