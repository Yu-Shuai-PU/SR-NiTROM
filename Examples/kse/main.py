import math
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
cmap_name = 'bwr'
#%% # Instantiate KSE class and KSE time-stepper class

traj_path = "./trajectories/"
data_path = "./data/"
fig_path = "./figures/"
os.makedirs(traj_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)

fname_X_template = data_path + "X_template.npy"
fname_X_template_dx = data_path + "X_template_dx.npy"
fname_X_template_dxx = data_path + "X_template_dxx.npy"
# fname_traj_init = data_path + "traj_init_%03d.npy" # for initial condition of u
# fname_traj_init_fitted = data_path + "traj_init_fitted_%03d.npy" # for initial condition of u fitted
fname_traj = traj_path + "traj_%03d.npy" # for u
fname_traj_fitted = traj_path + "traj_fitted_%03d.npy" # for u fitted
# fname_weight_traj = traj_path + "weight_traj_%03d.npy" // we are not loading weights, instead we dynamically compute them as we adjusting the learning timespan of trajectory-based NiTROM objective functions
# fname_weight_shift_amount = traj_path + "weight_shift_amount_%03d.npy"
# fname_weight_shift_speed = traj_path + "weight_shift_speed_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy" # for du/dt
fname_deriv_fitted = traj_path + "deriv_fitted_%03d.npy" # for du/dt fitted
fname_shift_amount = traj_path + "shift_amount_%03d.npy" # for shifting amount
fname_shift_speed = traj_path + "shift_speed_%03d.npy" # for shifting speed
fname_time = traj_path + "time.npy"

#%% # Generate and save trajectory
n_traj = 9
amp = 1
amp_array = np.array([-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0]) * amp
pool_inputs = (MPI.COMM_WORLD, n_traj)
pool_kwargs = {'fname_time':fname_time, 'fname_traj':fname_traj,'fname_traj_fitted':fname_traj_fitted,
               'fname_X_template':fname_X_template, 'fname_X_template_dx':fname_X_template_dx,
               'fname_X_template_dxx':fname_X_template_dxx,
               'fname_deriv':fname_deriv,'fname_deriv_fitted':fname_deriv_fitted,
               'fname_shift_amount':fname_shift_amount,'fname_shift_speed':fname_shift_speed}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)
pool.load_data()


# region 0: Set up NiTROM training parameters
# Our policy is like: we find the POD bases from the entire training dataset,
# but we train the bases and tensors only on the trajectory segments truncated in time.
# It's a curriculum learning strategy in a way: learn on short-term dynamics first, then extend to long-term dynamics.

initialization = "POD-Galerkin" # "POD-Galerkin" or "Previous NiTROM"
# initialization = "Previous NiTROM"
NiTROM_coeff_version = "new" # "old" or "new" for loading the NiTROM coefficients before or after the latest training
# NiTROM_coeff_version = "old"
# training_objects = "tensors_and_bases"
# training_objects = "tensors" 
training_objects = "no_alternating"
manifold = "Grassmann" # "Grassmann" or "Stiefel" for Psi manifold
# manifold = "Stiefel"
weight_decay_rate = 1 # if weight_decay_rate is not 1, then the snapshots at larger times will be given less weights in the cost function
r = 10 # ROM dimension
initial_relative_weight_c = 0.5 # as the training iterations go on, we will gradually increase the weight of shift amount term in the cost function
final_relative_weight_c = 0.5  # the final relative weight
sigmoid_steepness_c_weight = 2.0
initial_relative_weight_cdot = 0.05
final_relative_weight_cdot = 0.05
sigmoid_steepness_cdot_weight = 2.0
k0 = 0
kouter = 50
kinner_basis = 3
kinner_tensor = 3
initial_step_size_basis = 1e-1
initial_step_size_tensor = 1e-3
initial_step_size = 1e-3
initial_step_size_decrease_rate = 0.9
sufficient_decrease_rate_basis = 1e-3
sufficient_decrease_rate_tensor = 1e-3
initial_sufficient_decrease_rate = 1e-3
sufficient_decrease_rate_decay = 0.99
contraction_factor_tensor = 0.6
contraction_factor_basis = 0.6
contraction_factor = 0.6
snapshot_start_time_POD = 0
snapshot_end_time_POD = 1 + int(4 * (pool.n_snapshots - 1) // 4) # 1001
snapshot_start_time_NiTROM_training = 0
snapshot_end_time_NiTROM_training = 1 + int(0.1 * (pool.n_snapshots - 1) // 4)

# endregion

max_iterations = 20
leggauss_deg = 5
nsave_rom = 11 # nsave_rom = 1 + int(dt_sample/dt) = 1 + sample_interval

if pool.rank != 0:
    sys.stdout = open(os.devnull, 'w')

which_trajs = np.arange(0, pool.my_n_traj, 1)
which_times = np.arange(snapshot_start_time_POD,snapshot_end_time_POD,1)
L = 2 * np.pi
nx = 40
x = np.linspace(0, L, num=nx, endpoint=False)
nu = 4/87

poly_comp = [1, 2] # polynomial degree for the ROM dynamics

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

Phi_POD, cumulative_energy_proportion = opinf_fun.perform_POD(pool,opt_obj,r)
Psi_POD = Phi_POD.copy()
PhiF_POD = Phi_POD@scipy.linalg.inv(Psi_POD.T@Phi_POD)

fom = fom_class_kse.KSE(L, nu, nx, pool.X_template, pool.X_template_dx)

which_trajs = np.arange(0, pool.my_n_traj, 1)
which_times = np.arange(snapshot_start_time_NiTROM_training,snapshot_end_time_NiTROM_training,1)

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj_kwargs = {
    'X_template_dx': pool.X_template_dx,
    'X_template_dxx': pool.X_template_dxx,
    'spatial_deriv_method': fom.spatial_deriv,
    'inner_product_method': fom.inner_product,
    'outer_product_method': fom.outer_product}
opt_obj = classes.optimization_objects(*opt_obj_inputs, **opt_obj_kwargs)

Tensors_POD = fom.assemble_petrov_galerkin_tensors(Phi_POD, Psi_POD) # A, B, p, Q, s, M

fname_Phi_POD = data_path + "Phi_POD.npy"
np.save(fname_Phi_POD,Phi_POD)
fname_Psi_POD = data_path + "Psi_POD.npy"
np.save(fname_Psi_POD,Psi_POD)
fname_Tensors_POD = data_path + "Tensors_POD_Galerkin.npz"
np.savez(fname_Tensors_POD, *Tensors_POD)

# region 1: SR-Galerkin ROM

fname_traj_SRG = traj_path + "traj_SRG_%03d.npy" # for u
fname_traj_fitted_SRG = traj_path + "traj_fitted_SRG_%03d.npy"
fname_shift_amount_SRG = traj_path + "shift_amount_SRG_%03d.npy" # for shifting amount
fname_shift_speed_SRG = traj_path + "shift_speed_SRG_%03d.npy"
fname_relative_error_SRG = traj_path + "relative_error_SRG_%03d.npy"
fname_traj_FOM = traj_path + "traj_FOM_%03d.npy" # for u
fname_traj_fitted_FOM = traj_path + "traj_fitted_FOM_%03d.npy"
fname_shift_amount_FOM = traj_path + "shift_amount_FOM_%03d.npy" # for shifting amount
fname_shift_speed_FOM = traj_path + "shift_speed_FOM_%03d.npy"
fname_time_reconstruction = traj_path + "time_reconstruction.npy" # for plotting, in case our training & reconstruction timespan is shorter than the entire simulation timespan

np.save(fname_time_reconstruction,opt_obj.time)

relative_error = np.zeros(opt_obj.n_snapshots)
relative_error_space_time_SRG = np.zeros(pool.my_n_traj)

for k in range(pool.my_n_traj):
    traj_idx = k + pool.disps[pool.rank]
    print("Preparing SR-Galerkin simulation %d/%d"%(traj_idx,pool.n_traj))
    z0 = Psi_POD.T@opt_obj.X_fitted[k,:,0].reshape(-1)
    c0 = opt_obj.c[k,0]

    sol = solve_ivp(opt_obj.evaluate_rom_rhs,
                                          [opt_obj.time[0],opt_obj.time[-1]],
                                          np.hstack((z0, c0)),
                                          'RK45',
                                          t_eval=opt_obj.time,
                                          args=(np.zeros(r),) + Tensors_POD).y
    
    X_fitted_SRG = PhiF_POD@sol[:-1,:]
    c_SRG = sol[-1,:]
    X_SRG = np.zeros_like(X_fitted_SRG)
    cdot_SRG = np.zeros_like(c_SRG)

    for j in range (len(opt_obj.time)):
        X_SRG[:,j] = fom.shift(X_fitted_SRG[:,j], c_SRG[j])
        cdot_SRG[j] = opt_obj.compute_shift_speed(sol[:-1,j], Tensors_POD)
        relative_error[j] = np.linalg.norm(opt_obj.X[k,:,j] - X_SRG[:,j]) / np.linalg.norm(opt_obj.X[k,:,j])
        
    relative_error_space_time_SRG[k] = np.linalg.norm(opt_obj.X[k,:,:] - X_SRG)/np.linalg.norm(opt_obj.X[k,:,:])
    X_FOM = opt_obj.X[k,:,:]
    c_FOM = opt_obj.c[k,:]
    cdot_FOM = opt_obj.cdot[k,:]
    X_fitted_FOM = opt_obj.X_fitted[k,:,:]
        
    np.save(fname_traj_FOM%traj_idx,X_FOM)
    np.save(fname_traj_fitted_FOM%traj_idx,X_fitted_FOM)
    np.save(fname_shift_amount_FOM%traj_idx,c_FOM)
    np.save(fname_shift_speed_FOM%traj_idx,cdot_FOM)    
    np.save(fname_traj_SRG%traj_idx,X_SRG)
    np.save(fname_traj_fitted_SRG%traj_idx,X_fitted_SRG)
    np.save(fname_shift_amount_SRG%traj_idx,c_SRG)
    np.save(fname_shift_speed_SRG%traj_idx,cdot_SRG)
    np.save(fname_relative_error_SRG%traj_idx,relative_error)
    
    ### Plotting, things to be done:
    ### 1. switch from contourf to pcolormesh
    ### 2. apply Fourier spectral interpolation to plot a 40-mode solution on a 256-point grid for better visualization
    
    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,opt_obj.X[k,:,:].T, levels = np.linspace(-16, 16, 9), cmap=cmap_name)
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    if traj_idx == 0:
        plt.title(f"FOM solution, initial condition = uIC")
    else:
        plt.title(f"FOM solution, initial condition = uIC + {amp_array[traj_idx - 1]} * sin(x)")
    plt.savefig(fig_path + "traj_FOM_%03d.png"%traj_idx)
    plt.close()

    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,X_SRG.T, levels = np.linspace(-16, 16, 9), cmap=cmap_name)
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    if traj_idx == 0:
        plt.title(f"SRG solution, error: {relative_error_space_time_SRG[k]:.4e}, initial condition = uIC")
    else:
        plt.title(f"SRG solution, error: {relative_error_space_time_SRG[k]:.4e}, initial condition = uIC + {amp_array[traj_idx - 1]} * sin(x)")
    plt.savefig(fig_path + "traj_SRG_%03d.png"%traj_idx)
    plt.close()

# endregion

# region 2: SR-NiTROM ROM

Gr_Phi = manifolds.Grassmann(nx, r)
if manifold == "Stiefel":
    Gr_Psi = manifolds.Stiefel(nx, r)
elif manifold == "Grassmann":
    Gr_Psi = manifolds.Grassmann(nx, r)
Euc_A  = manifolds.Euclidean(r, r)
Euc_B  = manifolds.Euclidean(r, r, r)
Euc_p  = manifolds.Euclidean(r)
Euc_Q  = manifolds.Euclidean(r, r)

M = manifolds.Product([Gr_Phi, Gr_Psi, Euc_A, Euc_B, Euc_p, Euc_Q])
cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)

# Choose between POD-Galerkin initialization and previous training results
if initialization == "POD-Galerkin":
    print("Loading POD-Galerkin results as initialization")
    point = (Phi_POD, Psi_POD) + Tensors_POD[:-2]
    fname_Phi_NiTROM_old = data_path + "Phi_NiTROM_old.npy"
    np.save(fname_Phi_NiTROM_old,Phi_POD)
    fname_Psi_NiTROM_old = data_path + "Psi_NiTROM_old.npy"
    np.save(fname_Psi_NiTROM_old,Psi_POD)
    fname_Tensors_NiTROM_old = data_path + "Tensors_NiTROM_old.npz"
    np.savez(fname_Tensors_NiTROM_old, *Tensors_POD)

elif initialization == "Previous NiTROM":
    print("Loading previous NiTROM results as initialization (for curriculum learning)")
    if NiTROM_coeff_version == "new":
        Phi_NiTROM = np.load(data_path + "Phi_NiTROM.npy")
        if Phi_NiTROM.shape[1] != r:
            raise ValueError("The loaded Phi_NiTROM has shape %s, which does not match the specified r = %d"%(str(Phi_NiTROM.shape), r))
        
        Psi_NiTROM = np.load(data_path + "Psi_NiTROM.npy")
        npzfile = np.load(data_path + "Tensors_NiTROM.npz")
        Tensors_NiTROM = (
            npzfile['arr_0'],
            npzfile['arr_1'],
            npzfile['arr_2'],
            npzfile['arr_3'],
            npzfile['arr_4'],
            npzfile['arr_5']
        )
        point = (Phi_NiTROM, Psi_NiTROM) + Tensors_NiTROM[:-2]
        fname_Phi_NiTROM_old = data_path + "Phi_NiTROM_old.npy"
        np.save(fname_Phi_NiTROM_old,Phi_NiTROM)
        fname_Psi_NiTROM_old = data_path + "Psi_NiTROM_old.npy"
        np.save(fname_Psi_NiTROM_old,Psi_NiTROM)
        fname_Tensors_NiTROM_old = data_path + "Tensors_NiTROM_old.npz"
        np.savez(fname_Tensors_NiTROM_old, *Tensors_NiTROM)
    elif NiTROM_coeff_version == "old":
        Phi_NiTROM = np.load(data_path + "Phi_NiTROM_old.npy")
        if Phi_NiTROM.shape[1] != r:
            raise ValueError("The loaded Phi_NiTROM has shape %s, which does not match the specified r = %d"%(str(Phi_NiTROM.shape), r))
        
        Psi_NiTROM = np.load(data_path + "Psi_NiTROM_old.npy")
        npzfile = np.load(data_path + "Tensors_NiTROM_old.npz")
        Tensors_NiTROM = (
            npzfile['arr_0'],
            npzfile['arr_1'],
            npzfile['arr_2'],
            npzfile['arr_3'],
            npzfile['arr_4'],
            npzfile['arr_5']
        )
        point = (Phi_NiTROM, Psi_NiTROM) + Tensors_NiTROM[:-2]

if k0 == 0:
    costvec_NiTROM = []
    gradvec_NiTROM = []

for k in range(k0, k0 + kouter):
    
    relative_weight_c = update_relative_weights(initial_relative_weight_c, final_relative_weight_c, sigmoid_steepness_c_weight, k - k0, kouter)
    relative_weight_cdot = update_relative_weights(initial_relative_weight_cdot, final_relative_weight_cdot, sigmoid_steepness_cdot_weight, k - k0, kouter)
    sufficient_decrease_rate = initial_sufficient_decrease_rate * (sufficient_decrease_rate_decay ** (k - k0))
    step_size = initial_step_size * (initial_step_size_decrease_rate ** (k - k0))
    
    if pool.rank == 0:
        print("NiTROM training iteration %d/%d, progress: %.2f, relative weight c: %.4e, relative weight cdot: %.4e, sufficient decrease rate: %.3e"%(k+1, k0 + kouter, (k - k0) / kouter * 100, relative_weight_c, relative_weight_cdot, sufficient_decrease_rate))

    if training_objects == "tensors_and_bases":

        if np.mod(k, 2) == 0:
            which_fix = 'fix_bases'
            inner_iter = kinner_tensor
            line_searcher = myAdaptiveLineSearcher(contraction_factor = contraction_factor_tensor, # how much to reduce the step size in each iteration
                                            sufficient_decrease = sufficient_decrease_rate_tensor, # how much decrease is enough to accept the step
                                            max_iterations = max_iterations,
                                            initial_step_size = initial_step_size_tensor)
        else:
            which_fix = 'fix_tensors'
            inner_iter = kinner_basis
            line_searcher = myAdaptiveLineSearcher(contraction_factor = contraction_factor_basis, # how much to reduce the step size in each iteration
                                            sufficient_decrease = sufficient_decrease_rate_basis, # how much decrease is enough to accept the step
                                            max_iterations = max_iterations,
                                            initial_step_size = initial_step_size_basis)
    elif training_objects == "tensors":
        # for the control group, always optimize the tensors
        which_fix = 'fix_bases'
        inner_iter = kinner_tensor
        line_searcher = myAdaptiveLineSearcher(contraction_factor = contraction_factor_tensor, # how much to reduce the step size in each iteration
                                        sufficient_decrease = sufficient_decrease_rate_tensor, # how much decrease is enough to accept the step
                                        max_iterations = max_iterations,
                                        initial_step_size = initial_step_size_tensor)
        
    elif training_objects == "no_alternating":
        which_fix = 'fix_none'
        inner_iter = kinner_tensor + kinner_basis
        line_searcher = myAdaptiveLineSearcher(contraction_factor = contraction_factor, # how much to reduce the step size in each iteration
                                        sufficient_decrease = sufficient_decrease_rate, # how much decrease is enough to accept the step
                                        max_iterations = max_iterations,
                                        initial_step_size = step_size)
    
    opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
    opt_obj_kwargs = {
    'X_template_dx': pool.X_template_dx,
    'X_template_dxx': pool.X_template_dxx,
    'spatial_deriv_method': fom.spatial_deriv,
    'inner_product_method': fom.inner_product,
    'outer_product_method': fom.outer_product,
    'spatial_shift_method': fom.shift,
    'which_fix': which_fix,
    'relative_weight_c': relative_weight_c,
    'relative_weight_cdot': relative_weight_cdot,
    'weight_decay_rate': weight_decay_rate
    }
    opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
    
    print("Optimizing (%d/%d) with which_fix = %s"%(k+1,kouter,opt_obj.which_fix))
    
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
    print(cos_thetas)

    itervec_NiTROM_k = result.log["iterations"]["iteration"]
    costvec_NiTROM_k = result.log["iterations"]["cost"]
    gradvec_NiTROM_k = result.log["iterations"]["gradient_norm"]

    if k == 0:
        costvec_NiTROM.extend(costvec_NiTROM_k)
        gradvec_NiTROM.extend(gradvec_NiTROM_k)
    else:
        costvec_NiTROM.extend(costvec_NiTROM_k[1:])
        gradvec_NiTROM.extend(gradvec_NiTROM_k[1:])
        
# opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
# opt_obj_kwargs = {
# 'X_template_dx': pool.X_template_dx,
# 'X_template_dxx': pool.X_template_dxx,
# 'spatial_deriv_method': fom.spatial_deriv,
# 'inner_product_method': fom.inner_product,
# 'outer_product_method': fom.outer_product,
# 'spatial_shift_method': fom.shift,
# 'which_fix': which_fix,
# 'relative_weight_c': relative_weight_c,
# 'relative_weight_cdot': relative_weight_cdot,
# 'weight_decay_rate': weight_decay_rate
# }
# opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
# cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
# problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
# check_gradient(problem,x=point)

Phi_NiTROM, Psi_NiTROM = point[0:2]
Tensors_NiTROM_trainable = tuple(point[2:])
PhiF_NiTROM = Phi_NiTROM @ scipy.linalg.inv(Psi_NiTROM.T@Phi_NiTROM)
PhiF_NiTROM_dx = opt_obj.spatial_deriv(PhiF_NiTROM, order = 1)
cdot_denom_linear = np.zeros(r)
udx_linear = Psi_NiTROM.T @ PhiF_NiTROM_dx
u0_dx = opt_obj.X_template_dx

for i in range(r):
    cdot_denom_linear[i] = opt_obj.inner_product(u0_dx, PhiF_NiTROM_dx[:, i])

Tensors_NiTROM = Tensors_NiTROM_trainable + (cdot_denom_linear, udx_linear)

fname_Phi_NiTROM = data_path + "Phi_NiTROM.npy"
np.save(fname_Phi_NiTROM,Phi_NiTROM)
fname_Psi_NiTROM = data_path + "Psi_NiTROM.npy"
np.save(fname_Psi_NiTROM,Psi_NiTROM)
fname_Tensors_NiTROM = data_path + "Tensors_NiTROM.npz"
np.savez(fname_Tensors_NiTROM, *Tensors_NiTROM)

fname_traj_SRN = traj_path + "traj_SRN_%03d.npy" # for u
fname_traj_fitted_SRN = traj_path + "traj_fitted_SRN_%03d.npy"
fname_shift_amount_SRN = traj_path + "shift_amount_SRN_%03d.npy" # for shifting amount
fname_shift_speed_SRN = traj_path + "shift_speed_SRN_%03d.npy"
fname_relative_error_SRN = traj_path + "relative_error_SRN_%03d.npy"

relative_error = np.zeros(opt_obj.n_snapshots)
relative_error_space_time_SRN = np.zeros(pool.my_n_traj)

test_trial_consistency_percent = 1 - np.linalg.norm(Phi_NiTROM - Psi_NiTROM)/np.linalg.norm(Phi_NiTROM)
print("Test-trial difference of POD bases: %.4e%%"%(test_trial_consistency_percent*100))

for k in range(pool.my_n_traj):
    traj_idx = k + pool.disps[pool.rank]
    print("Preparing SR-NiTROM simulation %d/%d"%(traj_idx,n_traj))
    z0 = Psi_NiTROM.T@opt_obj.X_fitted[k,:,0].reshape(-1)
    c0 = opt_obj.c[k,0]

    sol = solve_ivp(opt_obj.evaluate_rom_rhs,
                                        [opt_obj.time[0],opt_obj.time[-1]],
                                        np.hstack((z0, c0)),
                                        'RK45',
                                        t_eval=opt_obj.time,
                                        args=(np.zeros(r),) + Tensors_NiTROM).y

    X_fitted_SRN = PhiF_NiTROM@sol[:-1,:]
    c_SRN = sol[-1,:]
    X_SRN = np.zeros_like(X_fitted_SRN)
    cdot_SRN = np.zeros_like(c_SRN)
    for j in range (len(opt_obj.time)):
        X_SRN[:,j] = fom.shift(X_fitted_SRN[:,j], c_SRN[j])
        cdot_SRN[j] = opt_obj.compute_shift_speed(sol[:-1,j], Tensors_NiTROM)
        relative_error[j] = np.linalg.norm(opt_obj.X[k,:,j] - X_SRN[:,j]) / np.linalg.norm(opt_obj.X[k,:,j])
        
    relative_error_space_time_SRN[k] = np.linalg.norm(opt_obj.X[k,:,:] - X_SRN)/np.linalg.norm(opt_obj.X[k,:,:])

    np.save(fname_traj_SRN%traj_idx,X_SRN)
    np.save(fname_traj_fitted_SRN%traj_idx,X_fitted_SRN)
    np.save(fname_shift_amount_SRN%traj_idx,c_SRN)
    np.save(fname_shift_speed_SRN%traj_idx,cdot_SRN)
    np.save(fname_relative_error_SRN%traj_idx,relative_error)

    plt.figure(figsize=(10,6))
    plt.contourf(x,opt_obj.time,X_SRN.T, levels = np.linspace(-16, 16, 9), cmap=cmap_name)
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.tight_layout()
    if traj_idx == 0:
        plt.title(f"SRN solution, error: {relative_error_space_time_SRN[k]:.4e}, initial condition = uIC")
    else:
        plt.title(f"SRN solution, error: {relative_error_space_time_SRN[k]:.4e}, initial condition = uIC + {amp_array[traj_idx - 1]} * sin(x)")
    plt.savefig(fig_path + "traj_SRN_%03d.png"%traj_idx)
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, opt_obj.c[k,:], color='k', linewidth=2, label='FOM')
    plt.plot(opt_obj.time, c_SRG, color='r', linewidth=2, label='SR-Galerkin ROM')
    plt.plot(opt_obj.time, c_SRN, color='b', linewidth=2, label='SR-NiTROM ROM')
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift amount")
    plt.ylim([-np.pi, np.pi])
    plt.legend()
    if traj_idx == 0:
        plt.title(f"Shift amount, initial condition = uIC")
    else:
        plt.title(f"Shift amount, initial condition = uIC + {amp_array[traj_idx - 1]} * sin(x)")
    plt.savefig(fig_path + "shift_amount_%03d.png"%traj_idx)
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(opt_obj.time, opt_obj.cdot[k,:], color='k', linewidth=2, label='FOM')
    plt.plot(opt_obj.time, cdot_SRG, color='r', linewidth=2, label='SR-Galerkin ROM')
    plt.plot(opt_obj.time, cdot_SRN, color='b', linewidth=2, label='SR-NiTROM ROM')
    plt.xlabel(r"$t$")
    plt.ylabel(r"Shift speed")
    plt.ylim([-2, 2])
    plt.legend()
    if traj_idx == 0:
        plt.title(f"Shift speed, initial condition = uIC")
    else:
        plt.title(f"Shift speed, initial condition = uIC + {amp_array[traj_idx - 1]} * sin(x)")
    plt.savefig(fig_path + "shift_speed_%03d.png"%traj_idx)
    plt.close()

if pool.rank == 0:
    print("Mean relative error of SR-Galerkin for all solutions: %.4e"%(np.mean(relative_error_space_time_SRG)))
    print("Mean relative error of SR-NiTROM for all solutions: %.4e"%(np.mean(relative_error_space_time_SRN)))

### plot the training error

plt.figure(figsize=(8,6))
# plt.semilogy(costvec_NiTROM,'-o',color=cOPT,label='SR-NiTROM')
plt.semilogy(costvec_NiTROM,'-o',color='blue',label='SR-NiTROM')
plt.xlabel('Iteration')
plt.ylabel('Cost function')
plt.title('Training error')
plt.legend()
plt.tight_layout()
plt.savefig(fig_path + f"training_error_NiTROM_start_time_{snapshot_start_time_NiTROM_training}_end_time_{snapshot_end_time_NiTROM_training}.png")
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
