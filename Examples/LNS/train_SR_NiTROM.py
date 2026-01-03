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
from func_plot import plot_ROM_vs_FOM

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
pool_inputs = (MPI.COMM_WORLD, params.n_traj)
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

# which_trajs = np.arange(0, pool.my_n_traj, 1)
# which_times = np.arange(params.snapshot_start_time_idx_POD,params.snapshot_end_time_idx_POD,1)

# opt_obj_inputs = (pool,which_trajs,which_times,params.leggauss_deg,params.nsave_rom,params.poly_comp)
# opt_obj = classes.optimization_objects(*opt_obj_inputs)

# Phi_POD, cumulative_energy_proportion = opinf_fun.perform_POD(pool,opt_obj,params.r,fom)
# Psi_POD = Phi_POD.copy() # Here, <Phi_POD, Phi_POD>_inner_product_3D = I, where inner_product_3D is the customized inner product defined in fom_class_LNS.py

# # To convert our customized inner product to standard L2 inner product, <q, q>_E = <Rq, Rq>_L2, we need to weight the projection bases:

# Psi_POD_w  = fom.apply_sqrt_inner_product_weight(Psi_POD) # R * Psi_POD
# Phi_POD_w  = Psi_POD_w.copy() # R * Phi_POD
# PhiF_POD_w = Phi_POD_w @ scipy.linalg.inv(Psi_POD_w.T @ Phi_POD_w) # PhiF_POD = Phi_POD (Psi_POD.T * W * Phi_POD)^{-1}, W = R^T * R

# print(f"relative difference between PhiF_POD and Phi_POD: {np.linalg.norm(PhiF_POD_w - Phi_POD_w) / np.linalg.norm(Phi_POD_w):.4e}") # should be very small, yes!
# ### Test the SR-Galerkin ROM simulation accuracy

# Tensors_POD_w = fom.assemble_weighted_petrov_galerkin_tensors(Psi_POD_w, PhiF_POD_w) # A, p, s, M

# fname_Phi_POD = params.data_path + "Phi_POD.npy"
# np.save(fname_Phi_POD,Phi_POD)
# fname_Psi_POD = params.data_path + "Psi_POD.npy"
# np.save(fname_Psi_POD,Psi_POD)
# fname_Psi_POD_weighted = params.data_path + "Psi_POD_weighted.npy"
# np.save(fname_Psi_POD_weighted,Psi_POD_w)
# fname_PhiF_POD_weighted = params.data_path + "Phi_POD_weighted.npy"
# np.save(fname_PhiF_POD_weighted,PhiF_POD_w)
# fname_Tensors_POD = params.data_path + "Tensors_POD_Galerkin_weighted.npz"
# np.savez(fname_Tensors_POD, *Tensors_POD_w)
# disturbance_kinetic_energy_FOM = np.zeros(opt_obj.n_snapshots)
# disturbance_kinetic_energy_SRG = np.zeros(opt_obj.n_snapshots)
# relative_error = np.zeros(opt_obj.n_snapshots)                                             
# relative_error_fitted = np.zeros(opt_obj.n_snapshots)
# relative_error_space_time_SRG = np.zeros(params.n_traj)


# for k in range(pool.my_n_traj):
#     traj_idx = k + pool.disps[pool.rank]
#     print("Preparing SR-Galerkin simulation %d/%d"%(traj_idx,pool.n_traj))
#     traj_SRG_init = Psi_POD_w.T@fom.apply_sqrt_inner_product_weight(opt_obj.X_fitted[k,:,0].reshape(-1))
#     shifting_amount_SRG_init = opt_obj.c[k,0]

#     sol_SRG = solve_ivp(opt_obj.evaluate_rom_rhs,
#                     [opt_obj.time[0],opt_obj.time[-1]],
#                     np.hstack((traj_SRG_init, shifting_amount_SRG_init)),
#                     'RK45',
#                     t_eval=opt_obj.time,
#                     args=(np.zeros(params.r),) + Tensors_POD_w).y
    
#     traj_fitted_SRG = fom.apply_inv_sqrt_inner_product_weight(PhiF_POD_w@sol_SRG[:-1,:])
#     traj_fitted_SRG_v = traj_fitted_SRG[0 : params.nx * params.ny * params.nz, :].reshape((params.nx, params.ny, params.nz, -1))
#     traj_fitted_SRG_eta = traj_fitted_SRG[params.nx * params.ny * params.nz : , :].reshape((params.nx, params.ny, params.nz, -1))
#     shifting_amount_SRG = sol_SRG[-1,:]
#     traj_SRG = np.zeros_like(traj_fitted_SRG)
#     shifting_speed_SRG = np.zeros_like(shifting_amount_SRG)

#     for j in range (len(opt_obj.time)):
#         traj_SRG_v_vec = fom.shift_x_input_3D(traj_fitted_SRG_v[:, :, :, j], shifting_amount_SRG[j])
#         traj_SRG_eta_vec = fom.shift_x_input_3D(traj_fitted_SRG_eta[:, :, :, j], shifting_amount_SRG[j])
#         traj_SRG[:,j] = np.concatenate((traj_SRG_v_vec.ravel(), traj_SRG_eta_vec.ravel()))
#         shifting_speed_SRG[j] = opt_obj.compute_shift_speed(sol_SRG[:-1,j], Tensors_POD_w)
#         diff_SRG_FOM_fitted = traj_fitted_SRG[:,j] - opt_obj.X_fitted[k,:,j]
#         diff_SRG_FOM        = traj_SRG[:,j] - opt_obj.X[k,:,j]
#         disturbance_kinetic_energy_FOM[j] = fom.inner_product_3D(opt_obj.X_fitted[k,0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
#                                                                opt_obj.X_fitted[k,params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)),
#                                                                opt_obj.X_fitted[k,0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
#                                                                opt_obj.X_fitted[k,params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)))
#         disturbance_kinetic_energy_SRG[j] = fom.inner_product_3D(traj_fitted_SRG[0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
#                                                                traj_fitted_SRG[params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)),
#                                                                traj_fitted_SRG[0 : params.nx * params.ny * params.nz,j].reshape((params.nx, params.ny, params.nz)),
#                                                                traj_fitted_SRG[params.nx * params.ny * params.nz : ,j].reshape((params.nx, params.ny, params.nz)))
#         relative_error_fitted[j] = fom.inner_product_3D(diff_SRG_FOM_fitted[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
#                                                  diff_SRG_FOM_fitted[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz)),
#                                                  diff_SRG_FOM_fitted[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
#                                                  diff_SRG_FOM_fitted[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz))) / disturbance_kinetic_energy_FOM[j]
#         relative_error[j] = fom.inner_product_3D(diff_SRG_FOM[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
#                                                  diff_SRG_FOM[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz)),
#                                                  diff_SRG_FOM[0 : params.nx * params.ny * params.nz].reshape((params.nx, params.ny, params.nz)),
#                                                  diff_SRG_FOM[params.nx * params.ny * params.nz : ].reshape((params.nx, params.ny, params.nz))) / disturbance_kinetic_energy_FOM[j]
        
#     traj_FOM = opt_obj.X[k,:,:]
#     shifting_amount_FOM = opt_obj.c[k,:]
#     shifting_speed_FOM = opt_obj.cdot[k,:]
#     traj_fitted_FOM = opt_obj.X_fitted[k,:,:]
#     traj_fitted_proj = Psi_POD_w.T @ fom.apply_sqrt_inner_product_weight(traj_fitted_FOM)
    
#     ### plotting
#     plot_ROM_vs_FOM(opt_obj, traj_idx, params.fig_path, relative_error, relative_error_fitted,
#                     disturbance_kinetic_energy_FOM, disturbance_kinetic_energy_SRG,
#                     shifting_amount_SRG, shifting_amount_FOM,
#                     shifting_speed_SRG, shifting_speed_FOM,
#                     traj_fitted_proj, sol_SRG,
#                     traj_FOM, traj_fitted_FOM, traj_SRG, traj_fitted_SRG, params.num_modes_to_plot,
#                     params.nx, params.ny, params.nz, params.dt, params.nsave, params.x, params.y, params.z, params.t_check_list, params.y_check)
    
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
    cos_thetas = np.linalg.svd(Q_Phi.T@Q_Psi, compute_uv=False)
    print([f"{x:.16f}" for x in cos_thetas])

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
PhiF_NiTROM_dx_w = opt_obj.take_derivative(PhiF_NiTROM_w, order = 1)
cdot_denom_linear = np.zeros(params.r)
udx_linear = Psi_NiTROM_w.T @ PhiF_NiTROM_dx_w
u0_dx_w = opt_obj.X_template_dx_weighted

for i in range(params.r):
    cdot_denom_linear[i] = np.dot(u0_dx_w, PhiF_NiTROM_dx_w[:, i])

Tensors_NiTROM = Tensors_NiTROM_trainable_w + (cdot_denom_linear, udx_linear)

fname_Phi_NiTROM = params.data_path + "Phi_NiTROM.npy"
np.save(fname_Phi_NiTROM,Phi_NiTROM)
fname_Psi_NiTROM = params.data_path + "Psi_NiTROM.npy"
np.save(fname_Psi_NiTROM,Psi_NiTROM)
fname_Phi_NiTROM_weighted = params.data_path + "Phi_NiTROM_weighted.npy"
np.save(fname_Phi_NiTROM_weighted,Phi_NiTROM_w)
fname_Psi_NiTROM_weighted = params.data_path + "Psi_NiTROM_weighted.npy"
np.save(fname_Psi_NiTROM_weighted,Psi_NiTROM_w)
fname_Tensors_NiTROM_weighted = params.data_path + "Tensors_NiTROM_weighted.npz"
np.savez(fname_Tensors_NiTROM_weighted, *Tensors_NiTROM)

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
