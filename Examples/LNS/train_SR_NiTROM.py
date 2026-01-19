import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI
import math

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
import configs
import classes
import nitrom_functions 
import opinf_functions as opinf_fun
import fom_class_LNS
from func_plot import plot_SRG_vs_FOM, plot_SRN_vs_FOM

def update_relative_weights(initial_relative_weight, final_relative_weight, sigmoid_steepness, k_relative, kouter):
    
    """
    This function is used to adjust dynamically the relative weights of different terms in the loss function as the iteration goes on.
    For example, at first we want to fix the solution profile error, but later we might want to emphasize more on the matching of shifting amounts
    """
    
    if kouter > 1:
        progress = k_relative / (kouter - 1)
    else:
        progress = 1.0

    sigmoid_variable = sigmoid_steepness * (progress - 0.5)
    
    tanh_val = math.tanh(sigmoid_variable)
    
    tanh_min = math.tanh(-sigmoid_steepness * 0.5)
    tanh_max = math.tanh(sigmoid_steepness * 0.5)
    f_progress = (tanh_val - tanh_min) / (tanh_max - tanh_min)

    if initial_relative_weight < 1e-6 or final_relative_weight < 1e-6:
        print("Both initial and final relative weights are zero, setting relative weight to zero directly.")
        relative_weight = 0.0
    else:
        relative_weight = math.exp((1.0 - f_progress) * math.log(initial_relative_weight) + f_progress * math.log(final_relative_weight))
    
    return relative_weight

# cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
# lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'


# region 1: Load parameters and FOM class

#%% # Instantiate KSE class and KSE time-stepper class
params = configs.load_configs()
fom = fom_class_LNS.LNS(params.Lx, params.Ly, params.Lz, 
                        params.nx, params.ny, params.nz,
                        params.y, params.Re,
                        params.U_base, params.U_base_dy, params.U_base_dyy)
tstep_kse_fom = fom_class_LNS.time_step_LNS(fom, params.time)

traj_template = np.load(params.fname_traj_template)
traj_template_dx = np.load(params.fname_traj_template_dx)
traj_template_dxx = np.load(params.fname_traj_template_dxx)
fom.load_template(traj_template, traj_template_dx)

# endregion

# region 2: Simulations
pool_inputs = (MPI.COMM_WORLD, params.n_traj_training)
pool_kwargs = {'fname_time':params.fname_time,
               'fname_X_template': params.fname_traj_template,
               'fname_X_template_dx':params.fname_traj_template_dx, 'fname_X_template_dx_weighted':params.fname_traj_template_dx_weighted,
               'fname_X_template_dxx':params.fname_traj_template_dxx, 'fname_X_template_dxx_weighted':params.fname_traj_template_dxx_weighted,
               'fname_traj':params.fname_traj, 'fname_traj_weighted':params.fname_traj_weighted,
               'fname_traj_fitted':params.fname_traj_fitted, 'fname_traj_fitted_weighted':params.fname_traj_fitted_weighted,
               'fname_deriv':params.fname_deriv,'fname_deriv_weighted':params.fname_deriv_weighted,
               'fname_deriv_fitted':params.fname_deriv_fitted,'fname_deriv_fitted_weighted':params.fname_deriv_fitted_weighted,
               'fname_shift_amount':params.fname_shift_amount,'fname_shift_speed':params.fname_shift_speed}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)
pool.load_data()

T_final = pool.time[-1]

# region 1: SR-POD-Galerkin ROM

which_trajs = np.arange(0, pool.my_n_traj, 1)
which_times = np.arange(params.snapshot_start_time_idx_POD,params.snapshot_end_time_idx_POD,1)

opt_obj_inputs = (pool,which_trajs,which_times,params.leggauss_deg,params.nsave_rom,params.poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

Phi_POD, cumulative_energy_proportion = opinf_fun.perform_POD(pool,opt_obj,params.r,fom)
Psi_POD = Phi_POD.copy() # Here, <Phi_POD, Phi_POD>_inner_product_3D = I, where inner_product_3D is the customized inner product defined in fom_class_LNS.py

for idx_mode in range(params.r):
    inner_product_mode = fom.inner_product_3D(Phi_POD[:params.nx * params.ny * params.nz,idx_mode].reshape((params.nx, params.ny, params.nz)),
                                              Phi_POD[params.nx * params.ny * params.nz:,idx_mode].reshape((params.nx, params.ny, params.nz)),
                                              Phi_POD[:params.nx * params.ny * params.nz,idx_mode].reshape((params.nx, params.ny, params.nz)),
                                              Phi_POD[params.nx * params.ny * params.nz:,idx_mode].reshape((params.nx, params.ny, params.nz))) 
    print(f"orthogonality check for mode {idx_mode}: {inner_product_mode:.4e}") # should be very close to 1.0


# To convert our customized inner product to standard L2 inner product, <q, q>_E = <Rq, Rq>_L2, we need to weight the projection bases:

Psi_POD_w  = fom.apply_sqrt_inner_product_weight(Psi_POD) # R * Psi_POD
Phi_POD_w  = Psi_POD_w.copy() # R * Phi_POD
print(f"orthogonality check (should be very small): {np.linalg.norm(Psi_POD_w.T @ Phi_POD_w - np.eye(params.r)):.4e}") # should be very small, yes!
PhiF_POD_w = Phi_POD_w @ scipy.linalg.inv(Psi_POD_w.T @ Phi_POD_w) # PhiF_POD = Phi_POD (Psi_POD.T * W * Phi_POD)^{-1}, W = R^T * R

print(f"relative difference between PhiF_POD and Phi_POD: {np.linalg.norm(PhiF_POD_w - Phi_POD_w) / np.linalg.norm(Phi_POD_w):.4e}") # should be very small, yes!
### Test the SR-Galerkin ROM simulation accuracy

Tensors_POD_w = fom.assemble_weighted_petrov_galerkin_tensors(Psi_POD_w, PhiF_POD_w) # A, p, s, M

fname_Phi_POD = params.data_path + "Phi_POD.npy"
np.save(fname_Phi_POD,Phi_POD)
fname_Psi_POD = params.data_path + "Psi_POD.npy"
np.save(fname_Psi_POD,Psi_POD)
fname_Psi_POD_weighted = params.data_path + "Psi_POD_weighted.npy"
np.save(fname_Psi_POD_weighted,Psi_POD_w)
fname_PhiF_POD_weighted = params.data_path + "Phi_POD_weighted.npy"
np.save(fname_PhiF_POD_weighted,PhiF_POD_w)
fname_Tensors_POD = params.data_path + "Tensors_POD_Galerkin_weighted.npz"
np.savez(fname_Tensors_POD, *Tensors_POD_w)

disturbance_kinetic_energy_FOM = np.zeros((params.n_traj_training, opt_obj.n_snapshots))
disturbance_kinetic_energy_SRG = np.zeros((params.n_traj_training, opt_obj.n_snapshots))
relative_error_SRG = np.zeros((params.n_traj_training, opt_obj.n_snapshots))                                             
relative_error_fitted_SRG = np.zeros((params.n_traj_training, opt_obj.n_snapshots))
traj_SRG = np.zeros((opt_obj.n_states, len(opt_obj.time))) # (n_states, n_snapshots)
shifting_speed_SRG = np.zeros((len(opt_obj.time))) # (n_snapshots)

for k in range(pool.my_n_traj):
    traj_idx = k + pool.disps[pool.rank]
    print("Preparing SR-Galerkin simulation %d/%d"%(traj_idx,pool.n_traj))
    traj_SRG_init = Psi_POD_w.T@opt_obj.X_fitted_weighted_init[k,:]
    shifting_amount_SRG_init = opt_obj.c[k,0]

    sol_SRG = solve_ivp(opt_obj.evaluate_rom_rhs,
                    [opt_obj.time[0],opt_obj.time[-1]],
                    np.hstack((traj_SRG_init, shifting_amount_SRG_init)),
                    'RK45',
                    t_eval=opt_obj.time,
                    args=(np.zeros(params.r),) + Tensors_POD_w).y
    
    traj_fitted_SRG = fom.apply_inv_sqrt_inner_product_weight(PhiF_POD_w@sol_SRG[:-1,:])
    traj_fitted_SRG_v = traj_fitted_SRG[0 : params.nx * params.ny * params.nz, :].reshape((params.nx, params.ny, params.nz, -1))
    traj_fitted_SRG_eta = traj_fitted_SRG[params.nx * params.ny * params.nz : , :].reshape((params.nx, params.ny, params.nz, -1))
    shifting_amount_SRG = sol_SRG[-1,:]
    
    traj_FOM, traj_fitted_FOM, traj_fitted_weighted_FOM = opt_obj.load_various_FOM_trajectories_idx(pool, k)

    for j in range (len(opt_obj.time)):
        traj_SRG_v_vec = fom.shift_x_input_3D(traj_fitted_SRG_v[:, :, :, j], shifting_amount_SRG[j])
        traj_SRG_eta_vec = fom.shift_x_input_3D(traj_fitted_SRG_eta[:, :, :, j], shifting_amount_SRG[j])
        traj_SRG[:,j] = np.concatenate((traj_SRG_v_vec.ravel(), traj_SRG_eta_vec.ravel()))
        shifting_speed_SRG[j] = opt_obj.compute_shift_speed(sol_SRG[:-1,j], Tensors_POD_w)
        diff_SRG_FOM_fitted = traj_fitted_SRG[:,j] - traj_fitted_FOM[:,j]
        diff_SRG_FOM        = traj_SRG[:,j] - traj_FOM[:,j]
        disturbance_kinetic_energy_FOM[traj_idx,j] = fom.inner_product_3D(traj_fitted_FOM[0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
                                                               traj_fitted_FOM[params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)),
                                                               traj_fitted_FOM[0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
                                                               traj_fitted_FOM[params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)))
        disturbance_kinetic_energy_SRG[traj_idx,j] = fom.inner_product_3D(traj_fitted_SRG[0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
                                                               traj_fitted_SRG[params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)),
                                                               traj_fitted_SRG[0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
                                                               traj_fitted_SRG[params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)))
        relative_error_fitted_SRG[traj_idx,j] = fom.inner_product_3D(diff_SRG_FOM_fitted[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRG_FOM_fitted[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRG_FOM_fitted[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRG_FOM_fitted[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz))) / disturbance_kinetic_energy_FOM[traj_idx,j]
        relative_error_SRG[traj_idx,j] = fom.inner_product_3D(diff_SRG_FOM[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRG_FOM[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRG_FOM[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRG_FOM[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz))) / disturbance_kinetic_energy_FOM[traj_idx,j]
     
    traj_fitted_FOM_proj_POD = Psi_POD_w.T @ traj_fitted_weighted_FOM
    
    plot_SRG_vs_FOM(opt_obj, traj_idx, params.fig_path_SRG, relative_error_SRG[traj_idx,:], relative_error_fitted_SRG[traj_idx,:],
                    disturbance_kinetic_energy_FOM[traj_idx,:], disturbance_kinetic_energy_SRG[traj_idx,:],
                    opt_obj.c[k,:], shifting_amount_SRG,
                    opt_obj.cdot[k,:], shifting_speed_SRG,
                    traj_fitted_FOM_proj_POD, sol_SRG[:-1,:],
                    traj_FOM, traj_SRG,
                    traj_fitted_FOM, traj_fitted_SRG,
                    params.num_modes_to_plot, params.nx, params.ny, params.nz, params.dt, params.nsave,
                    params.x, params.y, params.z, params.t_check_list_POD, params.y_check)
    
    np.save(f"{params.traj_path_FOM}disturbance_kinetic_energy_traj_{traj_idx:03d}.npy", disturbance_kinetic_energy_FOM[traj_idx,:])
    np.save(f"{params.traj_path_SRG}disturbance_kinetic_energy_traj_{traj_idx:03d}.npy", disturbance_kinetic_energy_SRG[traj_idx,:])
    np.save(f"{params.traj_path_SRG}relative_error_traj_{traj_idx:03d}.npy", relative_error_SRG[traj_idx,:])
    np.save(f"{params.traj_path_SRG}relative_error_fitted_traj_{traj_idx:03d}.npy", relative_error_fitted_SRG[traj_idx,:])
    np.save(f"{params.traj_path_SRG}shifting_amount_traj_{traj_idx:03d}.npy", shifting_amount_SRG)
    np.save(f"{params.traj_path_SRG}shifting_speed_traj_{traj_idx:03d}.npy", shifting_speed_SRG)
    np.save(f"{params.traj_path_SRG}traj_SRG_traj_{traj_idx:03d}.npy", traj_SRG)
    np.save(f"{params.traj_path_SRG}traj_fitted_SRG_traj_{traj_idx:03d}.npy", traj_fitted_SRG)
    np.save(f"{params.traj_path_SRG}sol_SRG_traj_{traj_idx:03d}.npy", sol_SRG)
    np.save(f"{params.traj_path_SRG}traj_fitted_FOM_proj_POD_traj_{traj_idx:03d}.npy", traj_fitted_FOM_proj_POD)
    
# endregion

# region 2: SR-NiTROM ROM
Gr_Phi_w = manifolds.Grassmann(2 * params.nx * params.ny * params.nz, params.r)
if params.manifold == "Stiefel":
    Gr_Psi_w = manifolds.Stiefel(2 * params.nx * params.ny * params.nz, params.r)
elif params.manifold == "Grassmann":
    Gr_Psi_w = manifolds.Grassmann(2 * params.nx * params.ny * params.nz, params.r)
Euc_A  = manifolds.Euclidean(params.r, params.r)
Euc_p  = manifolds.Euclidean(params.r)
M = manifolds.Product([Gr_Phi_w, Gr_Psi_w, Euc_A, Euc_p])

which_trajs = np.arange(0, pool.my_n_traj, 1)
which_times = np.arange(params.snapshot_start_time_idx_NiTROM_training,params.snapshot_end_time_idx_NiTROM_training,1)

opt_obj_inputs = (pool,which_trajs,which_times,params.leggauss_deg,params.nsave_rom,params.poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)
cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)

# Choose between POD-Galerkin initialization and previous training results
if params.initialization == "POD-Galerkin":
    print("Loading POD-Galerkin results as initialization")
    Phi_NiTROM_w = np.load(params.data_path + "Phi_POD_weighted.npy")
    if Phi_NiTROM_w.shape[1] != params.r:
        raise ValueError("The loaded Phi_NiTROM_weighted has shape %s, which does not match the specified r = %d"%(str(Phi_NiTROM_w.shape), params.r))
    Psi_NiTROM_w = np.load(params.data_path + "Psi_POD_weighted.npy")
    npzfile = np.load(params.data_path + "Tensors_POD_Galerkin_weighted.npz")
    Tensors_NiTROM_w = (
        npzfile['arr_0'],
        npzfile['arr_1'],
        npzfile['arr_2'],
        npzfile['arr_3']
    )   
    point = (Phi_NiTROM_w, Psi_NiTROM_w) + Tensors_NiTROM_w[:-2]
    fname_Phi_NiTROM_old = params.data_path + "Phi_NiTROM_weighted_old.npy"
    np.save(fname_Phi_NiTROM_old,Phi_NiTROM_w)
    fname_Psi_NiTROM_old = params.data_path + "Psi_NiTROM_weighted_old.npy"
    np.save(fname_Psi_NiTROM_old,Psi_NiTROM_w)
    fname_Tensors_NiTROM_old = params.data_path + "Tensors_NiTROM_weighted_old.npz"
    np.savez(fname_Tensors_NiTROM_old, *Tensors_NiTROM_w)  
    
elif params.initialization == "Previous NiTROM":
    print("Loading previous NiTROM results as initialization (for curriculum learning)")
    if params.NiTROM_coeff_version == "new":
        Phi_NiTROM_w = np.load(params.data_path + "Phi_NiTROM_weighted.npy")
        if Phi_NiTROM_w.shape[1] != params.r:
            raise ValueError("The loaded Phi_NiTROM_weighted has shape %s, which does not match the specified r = %d"%(str(Phi_NiTROM_w.shape), params.r))
        
        Psi_NiTROM_w = np.load(params.data_path + "Psi_NiTROM_weighted.npy")
        npzfile = np.load(params.data_path + "Tensors_NiTROM_weighted.npz")
        Tensors_NiTROM_w = (
            npzfile['arr_0'],
            npzfile['arr_1'],
            npzfile['arr_2'],
            npzfile['arr_3']
        )
        point = (Phi_NiTROM_w, Psi_NiTROM_w) + Tensors_NiTROM_w[:-2]
        fname_Phi_NiTROM_old = params.data_path + "Phi_NiTROM_weighted_old.npy"
        np.save(fname_Phi_NiTROM_old,Phi_NiTROM_w)
        fname_Psi_NiTROM_old = params.data_path + "Psi_NiTROM_weighted_old.npy"
        np.save(fname_Psi_NiTROM_old,Psi_NiTROM_w)
        fname_Tensors_NiTROM_old = params.data_path + "Tensors_NiTROM_weighted_old.npz"
        np.savez(fname_Tensors_NiTROM_old, *Tensors_NiTROM_w)
    elif params.NiTROM_coeff_version == "old":
        Phi_NiTROM_w = np.load(params.data_path + "Phi_NiTROM_weighted_old.npy")
        if Phi_NiTROM_w.shape[1] != params.r:
            raise ValueError("The loaded Phi_NiTROM_weighted has shape %s, which does not match the specified r = %d"%(str(Phi_NiTROM_w.shape), params.r))
        
        Psi_NiTROM_w = np.load(params.data_path + "Psi_NiTROM_weighted_old.npy")
        npzfile = np.load(params.data_path + "Tensors_NiTROM_weighted_old.npz")
        Tensors_NiTROM_w = (
            npzfile['arr_0'],
            npzfile['arr_1'],
            npzfile['arr_2'],
            npzfile['arr_3']
        )
        point = (Phi_NiTROM_w, Psi_NiTROM_w) + Tensors_NiTROM_w[:-2]
            
if params.k0 == 0:
    costvec_NiTROM = []
    gradvec_NiTROM = []
    
### precompute the weight for the norm, the shifting amount and the shifting speed
which_trajs = np.arange(0, pool.my_n_traj, 1)
which_times = np.arange(params.snapshot_start_time_idx_NiTROM_training,params.snapshot_end_time_idx_NiTROM_training,1)
opt_obj_inputs = (pool,which_trajs,which_times,params.leggauss_deg,params.nsave_rom,params.poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)
weight_traj, weight_shifting_amount, weight_shifting_speed = opt_obj.initialize_weights(fom)
    
for k in range(params.k0, params.k0 + params.kouter):
    
    relative_weight_c = update_relative_weights(params.initial_relative_weight_c, params.final_relative_weight_c, params.sigmoid_steepness_c_weight, k - params.k0, params.kouter)
    relative_weight_cdot = update_relative_weights(params.initial_relative_weight_cdot, params.final_relative_weight_cdot, params.sigmoid_steepness_cdot_weight, k - params.k0, params.kouter)
    
    if params.training_objects == "tensors_and_bases": # alternating optimization between bases and tensors

        if np.mod(k, 2) == 0:
            sufficient_decrease_rate = params.initial_sufficient_decrease_rate_tensor * (params.sufficient_decrease_rate_decay ** (k - params.k0))
            step_size = params.initial_step_size_tensor * (params.step_size_decrease_rate ** (k - params.k0))
    
            which_fix = 'fix_bases'
            inner_iter = params.kinner_tensor
            line_searcher = myAdaptiveLineSearcher(contraction_factor = params.contraction_factor_tensor, # how much to reduce the step size in each iteration
                                            sufficient_decrease = sufficient_decrease_rate, # how much decrease is enough to accept the step
                                            max_iterations = params.max_iterations,
                                            initial_step_size = step_size)
        else:
            sufficient_decrease_rate = params.initial_sufficient_decrease_rate_basis * (params.sufficient_decrease_rate_decay ** (k - params.k0))
            step_size = params.initial_step_size_basis * (params.step_size_decrease_rate ** (k - params.k0))
            which_fix = 'fix_tensors'
            inner_iter = params.kinner_basis
            line_searcher = myAdaptiveLineSearcher(contraction_factor = params.contraction_factor_basis, # how much to reduce the step size in each iteration
                                            sufficient_decrease = sufficient_decrease_rate, # how much decrease is enough to accept the step
                                            max_iterations = params.max_iterations,
                                            initial_step_size = step_size)
            
    elif params.training_objects == "tensors":
        # for the control group, always optimize the tensors with fixed bases
        sufficient_decrease_rate = params.initial_sufficient_decrease_rate_tensor * (params.sufficient_decrease_rate_decay ** (k - params.k0))
        step_size = params.initial_step_size_tensor * (params.step_size_decrease_rate ** (k - params.k0))
        which_fix = 'fix_bases'
        inner_iter = params.kinner_tensor
        line_searcher = myAdaptiveLineSearcher(contraction_factor = params.contraction_factor_tensor, # how much to reduce the step size in each iteration
                                        sufficient_decrease = sufficient_decrease_rate, # how much decrease is enough to accept the step
                                        max_iterations = params.max_iterations,
                                        initial_step_size = step_size)
        
    elif params.training_objects == "no_alternating": # optimize both bases and tensors in one shot together 
        which_fix = 'fix_none'
        inner_iter = params.kinner_tensor + params.kinner_basis
        sufficient_decrease_rate = params.initial_sufficient_decrease_rate * (params.sufficient_decrease_rate_decay ** (k - params.k0))
        step_size = params.initial_step_size * (params.step_size_decrease_rate ** (k - params.k0))
        line_searcher = myAdaptiveLineSearcher(contraction_factor = params.contraction_factor, # how much to reduce the step size in each iteration
                                        sufficient_decrease = sufficient_decrease_rate, # how much decrease is enough to accept the step
                                        max_iterations = params.max_iterations,
                                        initial_step_size = step_size)
    
    if pool.rank == 0:
        print("NiTROM training iteration %d/%d, progress: %.2f, relative weight c: %.4e, relative weight cdot: %.4e, sufficient decrease rate: %.3e"%(k+1, params.k0 + params.kouter, (k - params.k0) / params.kouter * 100, relative_weight_c, relative_weight_cdot, sufficient_decrease_rate))
    
    opt_obj_inputs = (pool,which_trajs,which_times,params.leggauss_deg,params.nsave_rom,params.poly_comp)
    opt_obj_kwargs = {
    'spatial_deriv_method': fom.diff_x_basis,
    'spatial_shift_method': fom.shift_x_state,
    'which_fix': which_fix,
    'relative_weight_c': relative_weight_c,
    'relative_weight_cdot': relative_weight_cdot,
    'weight_decay_rate': params.weight_decay_rate
    }
    opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
    opt_obj.recalibrate_weights(weight_traj, weight_shifting_amount, weight_shifting_speed)
    
    print("Optimizing (%d/%d) with which_fix = %s"%(k+1,params.kouter,opt_obj.which_fix))
    
    cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
    problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
    optimizer = optimizers.ConjugateGradient(max_iterations=inner_iter,min_step_size=1e-20,max_time=3600,\
                                              line_searcher=line_searcher,log_verbosity=1,verbosity=2)
    result = optimizer.run(problem,initial_point=point)
    point = result.point

    Phi_NiTROM = point[0]
    Psi_NiTROM = point[1]

    ## Compute the difference between Phi and Psi using the method of principle angles

    Q_Phi = np.linalg.qr(Phi_NiTROM)[0]
    Q_Psi = np.linalg.qr(Psi_NiTROM)[0]
    cos_thetas = np.linalg.svd(Q_Phi.T @ Q_Psi, compute_uv=False)

    # --- 修改部分 ---
    # 直接打印最小值，保留16位小数
    print(f"Minimum cos theta: {np.min(cos_thetas):.16f}")
    print(f"Primary angle between subspaces (degrees): {np.arccos(np.clip(np.min(cos_thetas), -1.0, 1.0)) * 180 / np.pi:.4e}")
    # ----------------
    # print([f"{x:.16f}" for x in cos_thetas])
    print(f"relative difference between Phi_NiTROM and Psi_NiTROM: {np.linalg.norm(Phi_NiTROM - Psi_NiTROM) / np.linalg.norm(Phi_NiTROM):.4e}")

    itervec_NiTROM_k = result.log["iterations"]["iteration"]
    costvec_NiTROM_k = result.log["iterations"]["cost"]
    gradvec_NiTROM_k = result.log["iterations"]["gradient_norm"]

    if k == 0:
        costvec_NiTROM.extend(costvec_NiTROM_k)
        gradvec_NiTROM.extend(gradvec_NiTROM_k)
    else:
        costvec_NiTROM.extend(costvec_NiTROM_k[1:])
        gradvec_NiTROM.extend(gradvec_NiTROM_k[1:])
        
opt_obj_inputs = (pool,which_trajs,which_times,params.leggauss_deg,params.nsave_rom,params.poly_comp)
opt_obj_kwargs = {
'spatial_deriv_method': fom.diff_x_basis,
'spatial_shift_method': fom.shift_x_state,
'which_fix': which_fix,
'relative_weight_c': relative_weight_c,
'relative_weight_cdot': relative_weight_cdot,
'weight_decay_rate': params.weight_decay_rate
}
opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
opt_obj.recalibrate_weights(weight_traj, weight_shifting_amount, weight_shifting_speed)
cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
check_gradient(problem,x=point)

Phi_NiTROM_w, Psi_NiTROM_w = point[0:2]
Phi_NiTROM = fom.apply_inv_sqrt_inner_product_weight(Phi_NiTROM_w)
Psi_NiTROM = fom.apply_inv_sqrt_inner_product_weight(Psi_NiTROM_w)
Tensors_NiTROM_trainable_w = tuple(point[2:])
PhiF_NiTROM_w = Phi_NiTROM_w @ scipy.linalg.inv(Psi_NiTROM_w.T@Phi_NiTROM_w)
PhiF_NiTROM_dx_w = fom.diff_x_basis(PhiF_NiTROM_w, order = 1)
cdot_denom_linear = np.zeros(params.r)
udx_linear = Psi_NiTROM_w.T @ PhiF_NiTROM_dx_w
u0_dx_w = opt_obj.X_template_dx_weighted

for i in range(params.r):
    cdot_denom_linear[i] = np.dot(u0_dx_w, PhiF_NiTROM_dx_w[:, i])

Tensors_NiTROM_w = Tensors_NiTROM_trainable_w + (cdot_denom_linear, udx_linear)

fname_Phi_NiTROM = params.data_path + "Phi_NiTROM.npy"
np.save(fname_Phi_NiTROM,Phi_NiTROM)
fname_Psi_NiTROM = params.data_path + "Psi_NiTROM.npy"
np.save(fname_Psi_NiTROM,Psi_NiTROM)
fname_Phi_NiTROM_weighted = params.data_path + "Phi_NiTROM_weighted.npy"
np.save(fname_Phi_NiTROM_weighted,Phi_NiTROM_w)
fname_Psi_NiTROM_weighted = params.data_path + "Psi_NiTROM_weighted.npy"
np.save(fname_Psi_NiTROM_weighted,Psi_NiTROM_w)
fname_Tensors_NiTROM_weighted = params.data_path + "Tensors_NiTROM_weighted.npz"
np.savez(fname_Tensors_NiTROM_weighted, *Tensors_NiTROM_w)

disturbance_kinetic_energy_SRN = np.zeros((pool.my_n_traj, opt_obj.n_snapshots))
relative_error_SRN = np.zeros((params.n_traj, opt_obj.n_snapshots))
relative_error_fitted_SRN = np.zeros((params.n_traj, opt_obj.n_snapshots))
sol_SRN = np.zeros((pool.my_n_traj, params.r + 1, opt_obj.n_snapshots)) 
traj_SRN = np.zeros_like(opt_obj.X) # (my_n_traj, n_states, n_snapshots)
traj_fitted_SRN = np.zeros_like(opt_obj.X_fitted) # (my_n_traj, n_states, n_snapshots)
shifting_amount_SRN = np.zeros_like(opt_obj.c) # (my_n_traj, n_snapshots)
shifting_speed_SRN = np.zeros_like(opt_obj.cdot) # (my_n_traj, n_snapshots)
traj_fitted_FOM_proj_NiTROM = np.zeros((pool.my_n_traj, params.r, opt_obj.n_snapshots)) 

for k in range(pool.my_n_traj):
    traj_idx = k + pool.disps[pool.rank]
    print("Preparing SR-NiTROM simulation %d/%d"%(traj_idx,pool.n_traj))
    traj_SRN_init = Psi_NiTROM_w.T@opt_obj.X_fitted_weighted_init[k,:]
    shifting_amount_SRN_init = opt_obj.c[k,0]

    sol_SRN[k,:,:] = solve_ivp(opt_obj.evaluate_rom_rhs,
                    [opt_obj.time[0],opt_obj.time[-1]],
                    np.hstack((traj_SRN_init, shifting_amount_SRN_init)),
                    'RK45',
                    t_eval=opt_obj.time,
                    args=(np.zeros(params.r),) + Tensors_NiTROM_w).y
    
    traj_fitted_SRN[k,:,:] = fom.apply_inv_sqrt_inner_product_weight(PhiF_NiTROM_w@sol_SRN[k,:-1,:])
    traj_fitted_SRN_v = traj_fitted_SRN[k,0 : params.nx * params.ny * params.nz, :].reshape((params.nx, params.ny, params.nz, -1))
    traj_fitted_SRN_eta = traj_fitted_SRN[k,params.nx * params.ny * params.nz : , :].reshape((params.nx, params.ny, params.nz, -1))
    shifting_amount_SRN[k,:] = sol_SRN[k,-1,:]
    
    for j in range (len(opt_obj.time)):
        traj_SRN_v_vec = fom.shift_x_input_3D(traj_fitted_SRN_v[:, :, :, j], shifting_amount_SRN[k,j])
        traj_SRN_eta_vec = fom.shift_x_input_3D(traj_fitted_SRN_eta[:, :, :, j], shifting_amount_SRN[k,j])
        traj_SRN[k,:,j] = np.concatenate((traj_SRN_v_vec.ravel(), traj_SRN_eta_vec.ravel()))
        shifting_speed_SRN[k,j] = opt_obj.compute_shift_speed(sol_SRN[k,:-1,j], Tensors_NiTROM_w)
        diff_SRN_FOM_fitted = traj_fitted_SRN[k,:,j] - opt_obj.X_fitted[k,:,j]
        diff_SRN_FOM        = traj_SRN[k,:,j] - opt_obj.X[k,:,j]
        disturbance_kinetic_energy_FOM[k, j] = fom.inner_product_3D(opt_obj.X_fitted[k,0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
                                                               opt_obj.X_fitted[k,params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)),
                                                               opt_obj.X_fitted[k,0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
                                                               opt_obj.X_fitted[k,params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)))
        disturbance_kinetic_energy_SRN[k, j] = fom.inner_product_3D(traj_fitted_SRN[k,0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
                                                               traj_fitted_SRN[k,params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)),
                                                               traj_fitted_SRN[k,0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
                                                               traj_fitted_SRN[k,params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)))
        relative_error_fitted_SRN[k, j] = fom.inner_product_3D(diff_SRN_FOM_fitted[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRN_FOM_fitted[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRN_FOM_fitted[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRN_FOM_fitted[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz))) / disturbance_kinetic_energy_FOM[k, j]
        relative_error_SRN[k, j] = fom.inner_product_3D(diff_SRN_FOM[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRN_FOM[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRN_FOM[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
                                                 diff_SRN_FOM[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz))) / disturbance_kinetic_energy_FOM[k, j]
        
    traj_fitted_FOM_proj_NiTROM[k,:,:] = Psi_NiTROM_w.T @ opt_obj.X_fitted_weighted[k,:,:]
    ### plotting
    plot_SRN_vs_FOM(opt_obj, traj_idx, params.fig_path_SRN,
                    relative_error_SRG[traj_idx,:], relative_error_SRN[traj_idx,:],
                    relative_error_fitted_SRG[traj_idx,:], relative_error_fitted_SRN[traj_idx,:],
                    disturbance_kinetic_energy_FOM[k,:], disturbance_kinetic_energy_SRG[k,:], disturbance_kinetic_energy_SRN[k,:],
                    opt_obj.c[k,:], shifting_amount_SRG[k,:], shifting_amount_SRN[k,:],
                    opt_obj.cdot[k,:], shifting_speed_SRG[k,:], shifting_speed_SRN[k,:],
                    traj_fitted_FOM_proj_POD[k,:,:], traj_fitted_FOM_proj_NiTROM[k,:,:], sol_SRG[k,:-1,:], sol_SRN[k,:-1,:],
                    opt_obj.X[k,:,:], traj_SRG[k,:,:], traj_SRN[k,:,:],
                    opt_obj.X_fitted[k,:,:], traj_fitted_SRG[k,:,:], traj_fitted_SRN[k,:,:],
                    params.num_modes_to_plot, params.nx, params.ny, params.nz, params.dt, params.nsave,
                    params.x, params.y, params.z, params.t_check_list_SRN, params.y_check)

# plot the training error

plt.figure(figsize=(8,6))
# plt.semilogy(costvec_NiTROM,'-o',color=cOPT,label='SR-NiTROM')
plt.semilogy(costvec_NiTROM,'-o',color='blue',label='SR-NiTROM')
plt.xlabel('Iteration')
plt.ylabel('Cost function')
plt.title('Training error')
plt.legend()
plt.tight_layout()
plt.savefig(params.fig_path_SRN + f"training_error_NiTROM_start_time_{params.snapshot_start_time_idx_NiTROM_training}_end_time_{params.snapshot_end_time_idx_NiTROM_training}.png")
plt.close()

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

# endregion
