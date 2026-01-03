# params.py : store simulation parameters and constants
import numpy as np
from dataclasses import dataclass, field
from typing import List
import fom_class_LNS
import os

@dataclass
class SimConfigs:
    
    # Simulation parameters
    
    Lx: float = 48
    Ly: float = 2
    Lz: float = 24
    nx: int = 96
    ny: int = 65
    nz: int = 96
    Re: float = 3000
    T: float = 200
    dt: float = 0.5
    nsave: int = 2
    amp: float = 1.0
    small_amp: float = 0.0001
    medium_amp: float = 0.0699
    large_amp: float = 0.1398
    traj_path: str = "./trajectories/"
    data_path: str = "./data/"
    fig_path: str = "./figures/"
    
    # Training methods
    
    n_traj : int = 1 # number of trajectories used for training
    r : int = 40 # dimension of the ROM
    poly_comp: List[int] = field(default_factory=lambda: [1])
    initialization: str = "POD-Galerkin" # "POD-Galerkin" or "Previous NiTROM"
    NiTROM_coeff_version: str = "new" # "old" or "new" for loading the NiTROM coefficients before or after the latest training
    training_objects: str = "tensors_and_bases" # "tensors_and_bases" or "tensors" or "no_alternating"
    manifold: str = "Grassmann" # "Grassmann" or "Stiefel" for Psi manifold choice
    timespan_percentage_POD: float = 1.00 # percentage of the entire timespan used for POD (always use all snapshots for POD)
    timespan_percentage_NiTROM_training: float = 0.02 # percentage of the entire timespan used for NiTROM training
    

    # Parameters for relative weights between different terms in the cost function
    
    weight_decay_rate: float = 1 # if weight_decay_rate is not 1, then the snapshots at larger times will be given less weights in the cost function
    initial_relative_weight_c: float = 0.5 # as the training iterations go on, we will gradually increase the weight of shift amount term in the cost function
    final_relative_weight_c: float = 0.5  # the final relative weight
    sigmoid_steepness_c_weight: float = 2.0
    initial_relative_weight_cdot: float = 0.05
    final_relative_weight_cdot: float = 0.05
    sigmoid_steepness_cdot_weight: float = 2.0
    
    # Parameters for training
    
    k0: int = 0
    kouter: int = 5
    kinner_basis: int = 3
    kinner_tensor: int = 3
    
    initial_step_size_basis: float = 1e-1
    initial_step_size_tensor: float = 1e-4
    initial_step_size: float = 1e-3
    
    step_size_decrease_rate: float = 0.9
    
    initial_sufficient_decrease_rate_basis: float = 1e-3
    initial_sufficient_decrease_rate_tensor: float = 1e-3
    initial_sufficient_decrease_rate: float = 1e-3
    
    sufficient_decrease_rate_decay: float = 0.99
    
    contraction_factor_tensor: float = 0.6
    contraction_factor_basis: float = 0.6
    contraction_factor: float = 0.6
    
    # Parameters for solving the adjoint equations during optimization
    max_iterations: int = 20
    leggauss_deg: int = 5
    nsave_rom: int = 11 # nsave_rom = 1 + int(dt_sample/dt) = 1 + sample_interval

    def __post_init__(self):
        
        os.makedirs(self.traj_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.fig_path, exist_ok=True)
                
        self.x = np.linspace(0, self.Lx, num=self.nx, endpoint=False)
        self.y = np.cos(np.pi * np.linspace(0, self.ny - 1, num=self.ny) / (self.ny - 1))
        self.z = np.linspace(0, self.Lz, num=self.nz, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Base flow
        self.U_base = 1 - self.y**2
        self.U_base_dy = -2 * self.y
        self.U_base_dyy = -2 * np.ones_like(self.y)
        
        # 时间相关
        self.time = self.dt * np.linspace(0, int(self.T/self.dt), int(self.T/self.dt) + 1, endpoint=True)
        self.tsave = self.time[::self.nsave]
        
        self.t_check_list = np.linspace(0, self.T, num = int(self.T/10) + 1, endpoint=True).tolist()
        self.y_check = -0.56
        
        self.fname_traj_template = self.data_path + "traj_template.npy"
        self.fname_traj_template_dx = self.data_path + "traj_template_dx.npy"
        self.fname_traj_template_dx_weighted = self.data_path + "traj_template_dx_weighted.npy"
        self.fname_traj_template_dxx = self.data_path + "traj_template_dxx.npy"
        self.fname_traj_template_dxx_weighted = self.data_path + "traj_template_dxx_weighted.npy"
        self.fname_traj_init = self.data_path + "traj_init_%03d.npy" # for initial condition of u
        self.fname_traj_init_weighted = self.data_path + "traj_init_weighted_%03d.npy" # for initial condition of u
        self.fname_traj_init_fitted = self.data_path + "traj_init_fitted_%03d.npy" # for initial condition of u fitted
        self.fname_traj_init_fitted_weighted = self.data_path + "traj_init_fitted_weighted_%03d.npy" # for initial condition of u fitted
        self.fname_traj = self.traj_path + "traj_%03d.npy" # for u
        self.fname_traj_weighted = self.traj_path + "traj_weighted_%03d.npy" # for u
        self.fname_traj_fitted = self.traj_path + "traj_fitted_%03d.npy" # for u fitted
        self.fname_traj_fitted_weighted = self.traj_path + "traj_fitted_weighted_%03d.npy" # for u fitted
        self.fname_deriv = self.traj_path + "deriv_%03d.npy" # for du/dt
        self.fname_deriv_weighted = self.traj_path + "deriv_weighted_%03d.npy" # for du/dt
        self.fname_deriv_fitted = self.traj_path + "deriv_fitted_%03d.npy" # for du/dt fitted
        self.fname_deriv_fitted_weighted = self.traj_path + "deriv_fitted_weighted_%03d.npy" # for du/dt fitted
        self.fname_shift_amount = self.traj_path + "shift_amount_%03d.npy" # for shifting amount
        self.fname_shift_speed = self.traj_path + "shift_speed_%03d.npy" # for shifting speed
        self.fname_time = self.traj_path + "time.npy"
        
        self.num_modes_to_plot = self.r
        self.snapshot_start_time_idx_POD = 0
        self.snapshot_end_time_idx_POD = 1 + int(self.timespan_percentage_POD * (len(self.tsave) - 1)) # 1001
        self.snapshot_start_time_idx_NiTROM_training = 0
        self.snapshot_end_time_idx_NiTROM_training = 1 + int(self.timespan_percentage_NiTROM_training * (len(self.tsave) - 1))
        
def load_configs():
    return SimConfigs()
