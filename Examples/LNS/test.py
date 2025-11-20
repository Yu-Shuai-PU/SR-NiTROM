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
import fom_class_kse

# cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
# lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'

#%% # Instantiate KSE class and KSE time-stepper class

sol_path = "./solutions_testing/"
data_path = "./data_testing/"
fig_path = "./figures_testing/"
data_training_path = "./data/"
os.makedirs(sol_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)

fname_sol_template = data_path + "sol_template.npy"
fname_sol_template_dx = data_path + "sol_template_dx.npy"
fname_sol_template_dxx = data_path + "sol_template_dxx.npy"
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
#%% # Generate and save trajectory

n_sol_testing = 4
amp = 1.0
r = 16
poly_comp = [1, 2] # polynomial degree for the ROM dynamics
ic_descriptions = [
    "uIC + 0.6*cos(1x) + 0.8*sin(3x) (In-Dist)",
    f"uIC + {amp}*cos(5x) (OOD Freq)",
    f"uIC + 0.7*cos(2x) + 0.7*sin(5x) (OOD Mixed)",
    f"uIC + {amp}*2*sin(x)cos(4x) (OOD Nonlinear)"
]
pool_inputs = (MPI.COMM_WORLD, n_sol_testing)
pool_kwargs = {'fname_time':fname_time, 'fname_sol':fname_sol,'fname_sol_fitted':fname_sol_fitted,
               'fname_sol_template':fname_sol_template, 'fname_sol_template_dx':fname_sol_template_dx,
               'fname_sol_template_dxx':fname_sol_template_dxx,
               'fname_sol_init':fname_sol_init, 'fname_sol_init_fitted':fname_sol_init_fitted,
               'fname_rhs':fname_rhs,'fname_rhs_fitted':fname_rhs_fitted,
               'fname_shift_amount':fname_shift_amount,'fname_shift_speed':fname_shift_speed,
               'fname_weight_sol':fname_weight_sol,'fname_weight_shift_amount':fname_weight_shift_amount}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)
pool.load_data()

which_trajs = np.arange(0, pool.my_n_sol, 1)
which_times = np.arange(0, pool.n_snapshots, 1)

# # 2. 如果当前处理器的 rank 不是 0，则重定向其标准输出
if pool.rank != 0:
    # os.devnull 是一个特殊的、跨平台的文件路径，所有写入它的内容都会被丢弃
    # 我们打开它并将其设置为新的 sys.stdout
    sys.stdout = open(os.devnull, 'w')
    # 可选：如果你也想屏蔽错误信息，可以加上下面这行
    # sys.stderr = open(os.devnull, 'w')

L = 2 * np.pi
nx = 40
x = np.linspace(0, L, num=nx, endpoint=False)
nu = 4/87
leggauss_deg = 5
nsave_rom = 11

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

fom = fom_class_kse.KSE(L, nu, nx, pool.sol_template, pool.sol_template_dx)

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj_kwargs = {
    'sol_template_dx': pool.sol_template_dx,
    'sol_template_dxx': pool.sol_template_dxx,
    'spatial_derivative_method': fom.take_derivative,
    'inner_product_method': fom.inner_product,
    'outer_product_method': fom.outer_product}
opt_obj = classes.optimization_objects(*opt_obj_inputs, **opt_obj_kwargs)

# region 1: SR-Galerkin ROM

Phi_POD = np.load(data_training_path + "Phi_POD.npy")
Psi_POD = np.load(data_training_path + "Psi_POD.npy")
PhiF_POD = Phi_POD@scipy.linalg.inv(Psi_POD.T@Phi_POD)
Tensors_POD_Galerkin_file = np.load(data_training_path + "Tensors_POD_Galerkin.npz")
Tensors_POD = (
    Tensors_POD_Galerkin_file['arr_0'],
    Tensors_POD_Galerkin_file['arr_1'],
    Tensors_POD_Galerkin_file['arr_2'],
    Tensors_POD_Galerkin_file['arr_3'],
    Tensors_POD_Galerkin_file['arr_4'],
    Tensors_POD_Galerkin_file['arr_5'])

fname_sol_SR_POD_Galerkin = sol_path + "sol_SR_Galerkin_%03d.npy" # for u
fname_sol_fitted_SR_POD_Galerkin = sol_path + "sol_fitted_SR_Galerkin_%03d.npy"
fname_shift_amount_SR_POD_Galerkin = sol_path + "shift_amount_SR_Galerkin_%03d.npy" # for shifting amount
fname_shift_speed_SR_POD_Galerkin = sol_path + "shift_speed_SR_Galerkin_%03d.npy"

relative_error_all_sol = 0.0
relative_error_fitted_all_sol = 0.0

for k in range(pool.my_n_sol):
    sol_idx = k + pool.disps[pool.rank]
    print("Preparing SR-Galerkin simulation %d/%d"%(sol_idx,n_sol_testing))
    z_IC = Psi_POD.T@opt_obj.sol_fitted[k,:,0].reshape(-1)
    shift_amount_IC = opt_obj.shift_amount[k,0]

    output_SR_POD_Galerkin = solve_ivp(opt_obj.evaluate_rom_rhs,
                                          [opt_obj.time[0],opt_obj.time[-1]],
                                          np.hstack((z_IC, shift_amount_IC)),
                                          'RK45',
                                          t_eval=opt_obj.time,
                                          args=(np.zeros(r),) + Tensors_POD).y
    
    sol_fitted_SR_POD_Galerkin = PhiF_POD@output_SR_POD_Galerkin[:-1,:]
    shift_amount_SR_POD_Galerkin = output_SR_POD_Galerkin[-1,:]
    sol_SR_POD_Galerkin = np.zeros_like(sol_fitted_SR_POD_Galerkin)
    shift_speed_SR_POD_Galerkin = np.zeros_like(shift_amount_SR_POD_Galerkin)
    for j in range (len(opt_obj.time)):
        sol_SR_POD_Galerkin[:,j] = fom.shift(sol_fitted_SR_POD_Galerkin[:,j], shift_amount_SR_POD_Galerkin[j])
        shift_speed_SR_POD_Galerkin[j] = opt_obj.compute_shift_speed(output_SR_POD_Galerkin[:r,j], Tensors_POD)

    np.save(fname_sol_SR_POD_Galerkin%sol_idx,sol_SR_POD_Galerkin)
    np.save(fname_sol_fitted_SR_POD_Galerkin%sol_idx,sol_fitted_SR_POD_Galerkin)
    np.save(fname_shift_amount_SR_POD_Galerkin%sol_idx,shift_amount_SR_POD_Galerkin)
    np.save(fname_shift_speed_SR_POD_Galerkin%sol_idx,shift_speed_SR_POD_Galerkin)
    
    sol_FOM = opt_obj.sol[k,:,:]
    sol_fitted_FOM = opt_obj.sol_fitted[k,:,:]  
    shift_amount_FOM = opt_obj.shift_amount[k,:]
    shift_speed_FOM = opt_obj.shift_speed[k,:]
    
    relative_error = np.linalg.norm(sol_FOM - sol_SR_POD_Galerkin)/np.linalg.norm(sol_FOM)
    relative_error_fitted = np.linalg.norm(sol_fitted_FOM - sol_fitted_SR_POD_Galerkin)/np.linalg.norm(sol_fitted_FOM)

    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,sol_SR_POD_Galerkin.T, levels = np.linspace(-16, 16, 9))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    plt.title(f"Testing SRG solution, error: {relative_error:.4e}, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "sol_SRG_%03d.png"%sol_idx)
    plt.close()

    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,(sol_FOM - sol_SR_POD_Galerkin).T, levels = np.linspace(-16, 16, 9))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    plt.title(f"Testing FOM-SRG diff, error: {relative_error:.4e}, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "sol_SRG_FOM_diff_%03d.png"%sol_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,sol_fitted_SR_POD_Galerkin.T, levels = np.linspace(-16, 16, 9))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    plt.title(f"Testing Fitted SRG solution, fitted error: {relative_error_fitted:.4e}, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "sol_SRG_fitted_%03d.png"%sol_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,(sol_fitted_FOM - sol_fitted_SR_POD_Galerkin).T, levels = np.linspace(-16, 16, 9))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    plt.title(f"Testing Fitted SRG-FOM diff, fitted error: {relative_error_fitted:.4e}, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "sol_SRG_FOM_fitted_diff_%03d.png"%sol_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, shift_amount_FOM, color='k', linewidth=2, label='FOM')
    plt.plot(opt_obj.time, shift_amount_SR_POD_Galerkin, color='r', linewidth=2, label='SR-Galerkin ROM')
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift amount")
    plt.ylim([-np.pi, np.pi])
    plt.legend()
    plt.tight_layout()
    plt.title(f"Testing Shift amount, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "shift_amount_SRG_FOM_%03d.png"%sol_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, shift_speed_FOM, color='k', linewidth=2, label='FOM')
    plt.plot(opt_obj.time, shift_speed_SR_POD_Galerkin, color='r', linewidth=2, label='SR-Galerkin ROM')
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift speed")
    plt.ylim([-2, 2])
    plt.legend()
    plt.tight_layout()
    plt.title(f"Testing Shift speed, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "shift_speed_SRG_FOM_%03d.png"%sol_idx)
    plt.close()

relative_error_SRG_all_sol = np.mean(np.asarray(pool.comm.allgather(relative_error)))
relative_error_SRG_fitted_all_sol = np.mean(np.asarray(pool.comm.allgather(relative_error_fitted)))

# endregion

# region 2: SR-NiTROM ROM

Phi_NiTROM = np.load(data_training_path + "Phi_NiTROM.npy")
Psi_NiTROM = np.load(data_training_path + "Psi_NiTROM.npy")
PhiF_NiTROM = Phi_NiTROM @ scipy.linalg.inv(Psi_NiTROM.T@Phi_NiTROM)
Tensors_NiTROM_file = np.load(data_training_path + "Tensors_NiTROM.npz")
Tensors_NiTROM = (
    Tensors_NiTROM_file['arr_0'],
    Tensors_NiTROM_file['arr_1'],
    Tensors_NiTROM_file['arr_2'],
    Tensors_NiTROM_file['arr_3'],
    Tensors_NiTROM_file['arr_4'],
    Tensors_NiTROM_file['arr_5'])

fname_sol_SR_NiTROM = sol_path + "sol_SR_NiTROM_%03d.npy" # for u
fname_sol_fitted_SR_NiTROM = sol_path + "sol_fitted_SR_NiTROM_%03d.npy"
fname_shift_amount_SR_NiTROM = sol_path + "shift_amount_SR_NiTROM_%03d.npy" # for shifting amount
fname_shift_speed_SR_NiTROM = sol_path + "shift_speed_SR_NiTROM_%03d.npy"

relative_error_all_sol = 0.0
relative_error_fitted_all_sol = 0.0

test_trial_consistency_percent = 1 - np.linalg.norm(Phi_NiTROM - Psi_NiTROM)/np.linalg.norm(Phi_NiTROM)
print("Test-trial difference of POD bases: %.4e%%"%(test_trial_consistency_percent*100))

for k in range(pool.my_n_sol):
    sol_idx = k + pool.disps[pool.rank]
    print("Preparing SR-NiTROM simulation %d/%d"%(sol_idx,n_sol_testing))
    z_IC_NiTROM = Psi_NiTROM.T@opt_obj.sol_fitted[k,:,0].reshape(-1)
    shift_amount_IC = opt_obj.shift_amount[k,0]

    output_SR_NiTROM = solve_ivp(opt_obj.evaluate_rom_rhs,
                                        [opt_obj.time[0],opt_obj.time[-1]],
                                        np.hstack((z_IC_NiTROM, shift_amount_IC)),
                                        'RK45',
                                        t_eval=opt_obj.time,
                                        args=(np.zeros(r),) + Tensors_NiTROM).y

    sol_fitted_SR_NiTROM = PhiF_NiTROM@output_SR_NiTROM[:-1,:]
    shift_amount_SR_NiTROM = output_SR_NiTROM[-1,:]
    sol_SR_NiTROM = np.zeros_like(sol_fitted_SR_NiTROM)
    shift_speed_SR_NiTROM = np.zeros_like(shift_amount_SR_NiTROM)
    for j in range (len(opt_obj.time)):
        sol_SR_NiTROM[:,j] = fom.shift(sol_fitted_SR_NiTROM[:,j], shift_amount_SR_NiTROM[j])
        shift_speed_SR_NiTROM[j] = opt_obj.compute_shift_speed(output_SR_NiTROM[:r,j], Tensors_NiTROM)

    np.save(fname_sol_SR_NiTROM%sol_idx,sol_SR_NiTROM)
    np.save(fname_sol_fitted_SR_NiTROM%sol_idx,sol_fitted_SR_NiTROM)
    np.save(fname_shift_amount_SR_NiTROM%sol_idx,shift_amount_SR_NiTROM)
    np.save(fname_shift_speed_SR_NiTROM%sol_idx,shift_speed_SR_NiTROM)
    
    sol_FOM = opt_obj.sol[k,:,:]
    sol_fitted_FOM = opt_obj.sol_fitted[k,:,:]  
    shift_amount_FOM = opt_obj.shift_amount[k,:]
    
    relative_error = np.linalg.norm(sol_FOM - sol_SR_NiTROM)/np.linalg.norm(sol_FOM)
    relative_error_fitted = np.linalg.norm(sol_fitted_FOM - sol_fitted_SR_NiTROM)/np.linalg.norm(sol_fitted_FOM)
    # print("Relative error of SR-NiTROM: %.4e"%(relative_error))
    # print("Relative error of fitted SR-NiTROM: %.4e"%(relative_error_fitted))

    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,sol_SR_NiTROM.T, levels = np.linspace(-16, 16, 9))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    plt.title(f"Testing SRN solution, error: {relative_error:.4e}, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "sol_SRN_%03d.png"%sol_idx)
    plt.close()

    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,(sol_FOM - sol_SR_NiTROM).T, levels = np.linspace(-16, 16, 9))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    plt.title(f"Testing SRN-FOM diff, error: {relative_error:.4e}, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "sol_SRN_FOM_diff_%03d.png"%sol_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,sol_fitted_SR_NiTROM.T, levels = np.linspace(-16, 16, 9))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    plt.title(f"Testing Fitted SRN solution, error: {relative_error_fitted:.4e}, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "sol_SRN_fitted_%03d.png"%sol_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,(sol_fitted_FOM - sol_fitted_SR_NiTROM).T, levels = np.linspace(-16, 16, 9))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    plt.title(f"Testing Fitted SRN-FOM diff, error: {relative_error_fitted:.4e}, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "sol_SRN_FOM_fitted_diff_%03d.png"%sol_idx)
    plt.close()

    # 加载 SRG 结果以进行最终比较
    shift_amount_SR_POD_Galerkin = np.load(fname_shift_amount_SR_POD_Galerkin%sol_idx)
    shift_speed_SR_POD_Galerkin = np.load(fname_shift_speed_SR_POD_Galerkin%sol_idx)

    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, shift_amount_FOM, color='k', linewidth=2, label='FOM')
    plt.plot(opt_obj.time, shift_amount_SR_POD_Galerkin, color='r', linewidth=2, label='SR-Galerkin ROM')
    plt.plot(opt_obj.time, shift_amount_SR_NiTROM, color='b', linewidth=2, label='SR-NiTROM ROM')
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift amount")
    plt.ylim([-np.pi, np.pi])
    plt.legend()
    plt.title(f"Testing shift amount, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "shift_amount_SRN_FOM_%03d.png"%sol_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, shift_speed_FOM, color='k', linewidth=2, label='FOM')
    plt.plot(opt_obj.time, shift_speed_SR_POD_Galerkin, color='r', linewidth=2, label='SR-Galerkin ROM')
    plt.plot(opt_obj.time, shift_speed_SR_NiTROM, color='b', linewidth=2, label='SR-NiTROM ROM')
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift speed")
    plt.ylim([-2, 2])
    plt.legend()
    plt.title(f"Testing shift speed, IC = {ic_descriptions[sol_idx]}")
    plt.savefig(fig_path + "shift_speed_SRN_FOM_%03d.png"%sol_idx)
    plt.close()

relative_error_SRN_all_sol = np.mean(np.asarray(pool.comm.allgather(relative_error)))
relative_error_SRN_fitted_all_sol = np.mean(np.asarray(pool.comm.allgather(relative_error_fitted)))
if pool.rank == 0:
    print("Mean relative error of SR-Galerkin for all solutions: %.4e"%(relative_error_SRG_all_sol))
    print("Mean relative error of fitted SR-Galerkin for all solutions: %.4e"%(relative_error_SRG_fitted_all_sol))
    print("Mean relative error of SR-NiTROM for all solutions: %.4e"%(relative_error_SRN_all_sol))
    print("Mean relative error of fitted SR-NiTROM for all solutions: %.4e"%(relative_error_SRN_fitted_all_sol))

# endregion
