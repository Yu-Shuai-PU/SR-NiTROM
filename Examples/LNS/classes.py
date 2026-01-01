import numpy as np 
from mpi4py import MPI
from itertools import combinations
from string import ascii_lowercase as ascii

class mpi_pool:

    def __init__(self,comm,n_traj,**kwargs):
        
        """ 
        This class contains all the info regarding the MPI pool that will be used 
        during optimization. It also loads the training data from disk. Every process
        (with Id "self.rank") owns its own instance of this class and its own chunk of 
        the training data. I.e., the whole training data set is distributed
        across the whole MPI pool.  
        
        comm:           MPI Communicator
        n_traj:         total number of trajectories we wish to load from disk
        fname_traj:     e.g., 'traj_%03d.npy' (string used to load each trajectory)
        fname_time:     e.g., 'time.txt' (time vector at which we save snapshots)
        
        Optional keyword arguments:
            fname_weights:          e.g., 'weight_%03d.npy' 
            fname_steady_forcing:   e.g., 'forcing_%03d.npy'
            fname_derivs:           e.g., 'fname_derivs_%03d.npy'
        """

        self.comm = comm                            # MPI communicator
        self.size = self.comm.Get_size()            # Total number of processes
        self.rank = self.comm.Get_rank()            # Id of the current process

        self.n_traj = n_traj                        # Total number of training trajectories
        if self.size > self.n_traj:
            raise ValueError ("You have more MPI processes than trajectories!")
        else:
            if self.rank == 0:
                print("Hello, you are running NiTROM with %d MPI processors."%self.size)

        self.my_n_traj = self.n_traj//self.size     # Number of trajectories owned by process self.rank
        self.my_n_traj += 1 if np.mod(self.n_traj,self.size) > self.rank else 0

        # Vectors used for future MPI communications
        self.counts = np.zeros(self.size,dtype=np.int64)    
        self.comm.Allgather([np.asarray([self.my_n_traj]),MPI.INT],[self.counts,MPI.INT])
        self.disps = np.concatenate(([0],np.cumsum(self.counts)[:-1]))
        
        # Load data from file
        self.kwargs = kwargs
        
    def load_data(self):
        self.time = np.load(self.kwargs.get('fname_time', None))
        self.load_template(self.kwargs)
        self.load_trajectories(self.kwargs)
        self.load_time_derivatives(self.kwargs)
        self.load_shift(self.kwargs)
        self.load_weights(self.kwargs)
        self.load_steady_forcing(self.kwargs)

    def load_trajectories(self,kwargs):
        
        fname_traj = kwargs.get('fname_traj',None)
        if fname_traj != None:
            self.fnames_traj = [fname_traj%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            X = [np.load(self.fnames_traj[k]) for k in range (self.my_n_traj)]
            self.n_snapshots = X[0].shape[1]
            
            self.X = np.zeros((self.my_n_traj,self.N,self.n_snapshots))
        
            for k in range (self.my_n_traj): self.X[k,] = X[k]
            
        fname_traj_weighted = kwargs.get('fname_traj_weighted',None)
        if fname_traj_weighted != None:
            self.fnames_traj_weighted = [fname_traj_weighted%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            X_weighted = [np.load(self.fnames_traj_weighted[k]) for k in range (self.my_n_traj)]
            
            self.X_weighted = np.zeros((self.my_n_traj,self.N,self.n_snapshots))

            for k in range (self.my_n_traj): self.X_weighted[k,] = X_weighted[k]
        
        fname_traj_fitted = kwargs.get('fname_traj_fitted',None)
        if fname_traj_fitted != None:
            self.fnames_traj_fitted = [fname_traj_fitted%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            X_fitted = [np.load(self.fnames_traj_fitted[k]) for k in range (self.my_n_traj)]
            
            self.X_fitted = np.zeros((self.my_n_traj,self.N,self.n_snapshots))

            for k in range (self.my_n_traj): self.X_fitted[k,] = X_fitted[k]
            
        fname_traj_fitted_weighted = kwargs.get('fname_traj_fitted_weighted',None)
        if fname_traj_fitted_weighted != None:
            self.fnames_traj_fitted_weighted = [fname_traj_fitted_weighted%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            X_fitted_weighted = [np.load(self.fnames_traj_fitted_weighted[k]) for k in range (self.my_n_traj)]
            
            self.X_fitted_weighted = np.zeros((self.my_n_traj,self.N,self.n_snapshots))

            for k in range (self.my_n_traj): self.X_fitted_weighted[k,] = X_fitted_weighted[k]
        
    def load_weights(self,kwargs):

        fname_weights_traj = kwargs.get('fname_weights_traj',None)
        self.weights_X = np.ones(self.my_n_traj)
        if fname_weights_traj != None:
            self.fnames_weights_traj = [fname_weights_traj%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            alpha = [np.load(self.fnames_weights_traj[k]) for k in range (self.my_n_traj)]
            self.weights_X = np.zeros(self.my_n_traj)
            for k in range (self.my_n_traj): self.weights_X[k] = alpha[k]
            
        fname_weights_shift_amount = kwargs.get('fname_weights_shift_amount',None)
        self.weights_c = np.ones(self.my_n_traj)
        if fname_weights_shift_amount != None:
            self.fnames_weights_shift_amount = [fname_weights_shift_amount%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            beta = [np.load(self.fnames_weights_shift_amount[k]) for k in range (self.my_n_traj)]
            self.weights_c = np.zeros(self.my_n_traj)
            for k in range (self.my_n_traj): self.weights_c[k] = beta[k]
            
        fname_weights_shift_speed = kwargs.get('fname_weights_shift_speed',None)
        self.weights_cdot = np.ones(self.my_n_traj)
        if fname_weights_shift_speed != None:
            self.fnames_weights_shift_speed = [fname_weights_shift_speed%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            gamma = [np.load(self.fnames_weights_shift_speed[k]) for k in range (self.my_n_traj)]
            self.weights_cdot = np.zeros(self.my_n_traj)
            for k in range (self.my_n_traj): self.weights_cdot[k] = gamma[k]
            
    def load_steady_forcing(self,kwargs):
        
        fname_forcing = kwargs.get('fname_steady_forcing',None)
        self.F = np.zeros((self.N, self.my_n_traj))
        if fname_forcing != None:
            self.fnames_forcing = [(fname_forcing)%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            for k in range (self.my_n_traj):  self.F[:,k] = np.load(self.fnames_forcing[k])
            
        fname_forcing_weighted = kwargs.get('fname_steady_forcing_weighted',None)
        self.F_weighted = np.zeros((self.N, self.my_n_traj))
        if fname_forcing_weighted != None:
            self.fnames_forcing_weighted = [(fname_forcing_weighted)%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            for k in range (self.my_n_traj):  self.F_weighted[:,k] = np.load(self.fnames_forcing_weighted[k])

    def load_time_derivatives(self,kwargs):
        
        fname_deriv = kwargs.get('fname_deriv',None)
        if fname_deriv != None:
            self.fnames_deriv = [fname_deriv%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            dX = [np.load(self.fnames_deriv[k]) for k in range (self.my_n_traj)]
            
            self.dX = np.zeros((self.my_n_traj,self.N,self.n_snapshots))
            
            for k in range (self.my_n_traj): self.dX[k,] = dX[k]
            
        fname_deriv_weighted = kwargs.get('fname_deriv_weighted',None)
        if fname_deriv_weighted != None:
            self.fnames_deriv_weighted = [fname_deriv_weighted%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            dX_weighted = [np.load(self.fnames_deriv_weighted[k]) for k in range (self.my_n_traj)]
            
            self.dX_weighted = np.zeros((self.my_n_traj,self.N,self.n_snapshots))
            
            for k in range (self.my_n_traj): self.dX_weighted[k,] = dX_weighted[k]
            
        fname_deriv_fitted = kwargs.get('fname_deriv_fitted',None)
        if fname_deriv_fitted != None:
            self.fnames_deriv_fitted = [fname_deriv_fitted%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            dX_fitted = [np.load(self.fnames_deriv_fitted[k]) for k in range (self.my_n_traj)]
            
            self.dX_fitted = np.zeros((self.my_n_traj,self.N,self.n_snapshots))
            
            for k in range (self.my_n_traj): self.dX_fitted[k,] = dX_fitted[k]
            
        fname_deriv_fitted_weighted = kwargs.get('fname_deriv_fitted_weighted',None)
        if fname_deriv_fitted_weighted != None:
            self.fnames_deriv_fitted_weighted = [fname_deriv_fitted_weighted%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            dX_fitted_weighted = [np.load(self.fnames_deriv_fitted_weighted[k]) for k in range (self.my_n_traj)]
            
            self.dX_fitted_weighted = np.zeros((self.my_n_traj,self.N,self.n_snapshots))
            
            for k in range (self.my_n_traj): self.dX_fitted_weighted[k,] = dX_fitted_weighted[k]
        
    def load_template(self,kwargs):
        
        fname_X_template = kwargs.get('fname_X_template',None)
        self.X_template = np.load(fname_X_template)
        self.N = self.X_template.shape[0]
        
        fname_X_template_dx = kwargs.get('fname_X_template_dx',None)
        self.X_template_dx = np.load(fname_X_template_dx)
        
        fname_X_template_dx_weighted = kwargs.get('fname_X_template_dx_weighted',None)
        self.X_template_dx_weighted = np.load(fname_X_template_dx_weighted)
            
        fname_X_template_dxx = kwargs.get('fname_X_template_dxx',None)
        self.X_template_dxx = np.load(fname_X_template_dxx)
        
        fname_X_template_dxx_weighted = kwargs.get('fname_X_template_dxx_weighted',None)
        self.X_template_dxx_weighted = np.load(fname_X_template_dxx_weighted)
            
    def load_shift(self,kwargs):
        fname_shift_amount = kwargs.get('fname_shift_amount',None)
        if fname_shift_amount != None:
            self.fnames_shift_amount = [fname_shift_amount%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            c = [np.load(self.fnames_shift_amount[k]) for k in range (self.my_n_traj)]
            self.c = np.zeros((self.my_n_traj, self.n_snapshots))

            for k in range (self.my_n_traj): self.c[k,] = c[k]
            
        fname_shift_speed = kwargs.get('fname_shift_speed',None)
        if fname_shift_speed != None:
            self.fnames_shift_speed = [fname_shift_speed%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            cdot = [np.load(self.fnames_shift_speed[k]) for k in range (self.my_n_traj)]
            self.cdot = np.zeros((self.my_n_traj, self.n_snapshots))

            for k in range (self.my_n_traj): self.cdot[k,] = cdot[k]

class optimization_objects:

    def __init__(self,mpi_pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp,**kwargs):

        """ 
        This class contains the training data information that will get passed to pymanopt. 
        
        mpi_pool:       an instance of the mpi_pool class
        which_trajs:    array of integers to extract a subset of the trajectories contained in 
                        mpi_pool.X. Useful if we end up using stochastic gradient descent
        which_times:    arrays of integers to extract a subset of a trajectory owned by mpi_pool. Useful
                        if we want to start training on short trajectories and then progressively extend 
                        the length of the trajectories
        leggauss_deg:   number of Gauss-Legendre quadrature points used to approximate the integrals
                        in the gradient (see Prop. 2.1 in NiTROM arXiv paper)
        nsave_rom:      number of ROM snapshots to store in between two adjacent FOM snapshots
        poly_comp:      component of the polynomial ROM (e.g., [1,2] is a quadratic model with both first and
                                                         second order components) 

        Optional keyword arguments:
            which_fix:              one of fix_bases, fix_tensors or fix_none (default is fix_none)
            stab_promoting_pen:     value of L2 regularization coefficient
            stab_promoting_tf:      value of final time for stability promoting penalty
            stab_promoting_ic:      random (unit-norm) vector to probe the stability penalty
        """

        self.X = mpi_pool.X[which_trajs,:,:]
        self.X_weighted = mpi_pool.X_weighted[which_trajs,:,:]
        self.X_fitted = mpi_pool.X_fitted[which_trajs,:,:]
        self.X_fitted_weighted = mpi_pool.X_fitted_weighted[which_trajs,:,:]
        self.F = mpi_pool.F[:,which_trajs]
        self.F_weighted = mpi_pool.F_weighted[:,which_trajs]
        self.c = mpi_pool.c[which_trajs,:]
        self.cdot = mpi_pool.cdot[which_trajs,:]
        self.time = mpi_pool.time[which_times]

        self.X = self.X[:,:,which_times]
        self.X_weighted = self.X_weighted[:,:,which_times]
        self.X_fitted = self.X_fitted[:,:,which_times]
        self.X_fitted_weighted = self.X_fitted_weighted[:,:,which_times]
        self.c = self.c[:,which_times]
        self.cdot = self.cdot[:,which_times]
        
        self.my_n_traj, _, self.n_snapshots = self.X.shape
        self.leggauss_deg = leggauss_deg
        self.nsave_rom = nsave_rom
        self.poly_comp = poly_comp
        
        self.generate_einsum_subscripts_rhs_poly()
        self.generate_einsum_subscripts_rhs_shift_speed_numer()
        
        # Count the total number of trajectories in this batch and
        # scale the weight accordingly so that the cost function measures
        # the average error over snapshots and trajectories. (Notice that 
        # if all trajectories are loaded, then np.sum(counts) = mpi_pool.n_traj)
        self.counts = np.zeros(mpi_pool.size,dtype=np.int64)
        mpi_pool.comm.Allgather([np.asarray([self.my_n_traj]),MPI.INT],[self.counts,MPI.INT])
        
        # Parse the keyword arguments
        self.which_fix = kwargs.get('which_fix','fix_none')
        if self.which_fix not in ['fix_tensors','fix_bases','fix_none']:
            raise ValueError ("which_fix must be fix_none, fix_tensors or fix_bases")
        
        self.l2_pen = kwargs.get('stab_promoting_pen',None)
        self.pen_tf = kwargs.get('stab_promoting_tf',None)
        self.randic = kwargs.get('stab_promoting_ic',None)
        
        if self.l2_pen != None and self.pen_tf == None:
            raise ValueError ("If you provide a value for stab_promoting_pen you \
                              also have to provide a value for stab_promoting_tf")
                              
        if self.l2_pen != None and self.randic == None:
            raise ValueError ("If you provide a value for stab_promoting_pen you \
                              also have to provide a random ic vector of the same \
                              size as the ROM")
                              
        if self.l2_pen != None and 1 not in self.poly_comp:
            raise ValueError ("The penalty is currently implemented for the linear term \
                              in the rom dynamics. You have no linear term.")
                              
        if self.randic != None: 
            self.randic /= np.linalg.norm(self.randic)
            self.randic = self.randic.reshape(-1)
            
        self.relative_weight_c = kwargs.get('relative_weight_c',1.0)
        self.relative_weight_cdot = kwargs.get('relative_weight_cdot',1.0)
        self.weight_decay_rate = kwargs.get('weight_decay_rate',1.0)
        self.decay_factor = self.weight_decay_rate ** (np.arange(self.n_snapshots))
        self.decay_factor = self.decay_factor / np.sum(self.decay_factor)
        
        self.X_template_dx = mpi_pool.X_template_dx
        self.X_template_dx_weighted = mpi_pool.X_template_dx_weighted
        self.X_template_dxx = mpi_pool.X_template_dxx
        self.X_template_dxx_weighted = mpi_pool.X_template_dxx_weighted
        self.spatial_deriv = kwargs.get('spatial_deriv_method',None)
        self.shift         = kwargs.get('spatial_shift_method',None)
        
        self.state_mag_threshold = 1e4
        self.cdot_denom_threshold = 1e-15
        
    def initialize_weights(self, fom):
        
        weight_traj = np.zeros(self.my_n_traj)
        weight_shifting_amount = np.zeros(self.my_n_traj)
        weight_shifting_speed = np.zeros(self.my_n_traj)
        
        for idx in range(self.my_n_traj):
            weight_traj[idx] = 1.0/np.mean(np.linalg.norm(self.X_weighted[idx,:,:],axis=0)**2)
            weight_shifting_amount[idx] = 1.0/fom.Lx ** 2
            weight_shifting_speed[idx] = 1.0/np.mean((self.cdot[idx,:] - np.mean(self.cdot[idx,:]))**2)
            if np.max(weight_shifting_amount[idx]) > 1e8:
                raise Warning ("The shift amount for trajectory %d is very small, which makes weight very large (%.4e). \
                                Consider increasing it to avoid ill-conditioning or give up using symmetry reduction."%(idx+self.disps[mpi_pool.rank], self.weights_c[idx])) 
        
        return weight_traj, weight_shifting_amount, weight_shifting_speed
        
    def recalibrate_weights(self, weight_traj, weight_shifting_amount, weight_shifting_speed):
        
        self.weights_X = np.zeros((self.my_n_traj,self.n_snapshots))
        self.weights_c = np.zeros((self.my_n_traj,self.n_snapshots))
        self.weights_cdot = np.zeros((self.my_n_traj,self.n_snapshots))
                
        for idx in range(self.my_n_traj):
            self.weights_X[idx,:] = weight_traj[idx] * self.decay_factor
            self.weights_c[idx,:] = weight_shifting_amount[idx] * self.decay_factor
            self.weights_cdot[idx,:] = weight_shifting_speed[idx] * self.decay_factor
            if np.max(self.weights_c[idx]) > 1e8:
                raise Warning ("The shift amount for trajectory %d is very small, which makes weight very large (%.4e). \
                                Consider increasing it to avoid ill-conditioning or give up using symmetry reduction."%(idx+self.disps[mpi_pool.rank], self.weights_c[idx]))

        self.weights_X /= np.sum(self.counts)*self.n_snapshots
        self.weights_c *= self.relative_weight_c / (np.sum(self.counts)*self.n_snapshots)
        self.weights_cdot *= self.relative_weight_cdot / (np.sum(self.counts)*self.n_snapshots)    
        
    def generate_einsum_subscripts_rhs_poly(self):
        """
            Generates the indices for the einsum evaluation of the 
            dynamics
        """
        ss = []
        for k in self.poly_comp:
            ssk = ascii[:k+1]
            ssk = [ssk] + [s for s in ssk[1:]]
            ss.append(ssk)
        
        self.einsum_ss_rhs_poly = tuple(ss)
        
    def generate_einsum_subscripts_rhs_shift_speed_numer(self):
        """
            Generates the indices for the einsum evaluation of the 
            numerator of the reconstruction equation for computing the shift speed
        """
        ss = []
        for k in self.poly_comp:
            ssk = ascii[:k]
            ssk = [ssk] + [s for s in ssk]
            ss.append(ssk)
        
        self.einsum_ss_rhs_shift_speed_numer = tuple(ss)

    def evaluate_rom_rhs(self,t,zc,u,*operators,**kwargs):
        """
            Function that can be fed into scipys solve_ivp. 
            t:          time instance
            a:          state vector (reduced state + shift amount)
            u:          a steady forcing vector
            operators:  (A2,A3,A4,...)
            
            Optional keyword arguments:
                'forcing_interp':   a scipy interpolator f that gives us a forcing f(t)
        """
        z = zc[:-1]

        if np.linalg.norm(z) >= self.state_mag_threshold:
            dzdt = 0.0*z
            dcdt = 0.0
            # raise ValueError ("The norm of the state vector is too large!")
            print("Warning: The norm of the state vector is too large! Modified RHS to zero.")
        else:
            f = kwargs.get('forcing_interp',None)
            f = f(t) if f != None else np.zeros(len(z))
            u = u.copy() if hasattr(u,"__len__") == True else u(t)
            dzdt = u + f
            
            cdot_denom_linear = operators[-2]
            udx_linear = operators[-1]
            cdot_denom = np.einsum('i,i',cdot_denom_linear,z)
            # print("time = ", t, "cdot_denom", cdot_denom)
            # print("cdot_denom:", cdot_denom)
            if abs(cdot_denom) < self.cdot_denom_threshold:
                # raise ValueError ("Denominator in reconstruction equation of the shifting speed is too close to zero!")
                print("Warning: Denominator in reconstruction equation of the shifting speed is too close to zero! Modified shift speed to zero.")
                dcdt = 0.0
                # cdot_denom = 1e-2 * np.sign(cdot_denom)
                return np.hstack((dzdt, dcdt))
            else:
                udx = np.einsum('ij, j', udx_linear,z)

                cdot_numer = 0.0
                
                for (i, k) in enumerate(self.poly_comp):
                    equation = ",".join(self.einsum_ss_rhs_poly[i])
                    operands = [operators[i]] + [z for _ in range(k)]
                    dzdt += np.einsum(equation,*operands)
                    equation = ",".join(self.einsum_ss_rhs_shift_speed_numer[i])
                    operands = [operators[i + len(self.poly_comp)]] + [z for _ in range(k)]
                    cdot_numer -= np.einsum(equation,*operands)
                    
                dzdt += (cdot_numer/cdot_denom) * udx
                dcdt = cdot_numer/cdot_denom
                
        return np.hstack((dzdt, dcdt))
    
    def evaluate_rom_unreduced_rhs(self,t,z,u,*operators,**kwargs):
        """
            Function that can be fed into scipys solve_ivp. 
            t:          time instance
            a:          state vector (reduced state + shift amount)
            u:          a steady forcing vector
            operators:  (A2,A3,A4,...)
            
            Optional keyword arguments:
                'forcing_interp':   a scipy interpolator f that gives us a forcing f(t)
        """
        # print(f"evaluate_rom_unreduced_rhs called at time t = {t}, z_magnitude = {np.linalg.norm(z)}")

        if np.linalg.norm(z) >= self.state_mag_threshold:
            dzdt = 0.0*z
            # raise ValueError ("The norm of the state vector is too large!")
            print("Warning: The norm of the state vector is too large! Modified RHS to zero.")
        else:
            f = kwargs.get('forcing_interp',None)
            f = f(t) if f != None else np.zeros(len(z))
            u = u.copy() if hasattr(u,"__len__") == True else u(t)
            dzdt = u + f
            
            for (i, k) in enumerate(self.poly_comp):
                equation = ",".join(self.einsum_ss_rhs_poly[i])
                operands = [operators[i]] + [z for _ in range(k)]
                dzdt += np.einsum(equation,*operands)
                
        return dzdt

    def compute_shift_speed(self, z, operators):
        """
            Function to compute the shift speed given a state z and the ROM operators
        """
        cdot_denom_linear = operators[-2]
        cdot_denom = np.einsum('i,i',cdot_denom_linear,z)
        # print("cdot_denom:", cdot_denom)
        if abs(cdot_denom) < self.cdot_denom_threshold:
            # raise ValueError ("Denominator in reconstruction equation of the shifting speed is too close to zero!")
            print("Warning: Denominator in reconstruction equation of the shifting speed is too close to zero! Modified shift speed to zero.")
            return 0.0
        else:
            cdot_numer = 0.0
            
            for (i, k) in enumerate(self.poly_comp):
                equation = ",".join(self.einsum_ss_rhs_shift_speed_numer[i])
                operands = [operators[i + len(self.poly_comp)]] + [z for _ in range(k)]
                cdot_numer -= np.einsum(equation,*operands)
                
            # print("cdot:", cdot_numer/cdot_denom)
            return cdot_numer/cdot_denom
    
    def compute_shift_speed_numer(self, z, operators):
        """
            Function to compute the shift speed given a state z and the ROM operators
        """
        cdot_numer = 0.0
        
        for (i, k) in enumerate(self.poly_comp):
            equation = ",".join(self.einsum_ss_rhs_shift_speed_numer[i])
            operands = [operators[i + len(self.poly_comp)]] + [z for _ in range(k)]
            cdot_numer -= np.einsum(equation,*operands)

        return cdot_numer

    def compute_shift_speed_denom(self, z, operators):
        """
            Function to compute the denominator of the shift speed given a state z and the ROM operators
        """
        cdot_denom_linear = operators[-2]
        cdot_denom = np.einsum('i,i',cdot_denom_linear,z)
        if abs(cdot_denom) < self.cdot_denom_threshold:
            # raise ValueError ("Denominator in reconstruction equation of the shifting speed is too close to zero!")
            print("Warning: Denominator in reconstruction equation of the shifting speed is too close to zero! Modified denominator to small value.")
            return self.cdot_denom_threshold * np.sign(cdot_denom)
        else:
            return cdot_denom

    def evaluate_rom_adjoint(self,t,xi,fz,fcdot, const, PhiF_dx,Psi, *operators):
        """
            Function that can be fed into scipys solve_ivp. 
            t:          time instance
            xi:         state vector of the adjoint system
            fq:         interpolator (from scipy.interpolate) to evaluate the
                        base flow at time t
            operators:  (A, B, p, Q, s, M)
        """

        if np.linalg.norm(xi) >= self.state_mag_threshold:
            print("Warning: The norm of the adjoint state vector is too large! Modified RHS of the adjoint RHS to zero.")
            dxidt = 0.0*xi
        else:
            dxidt = 0.0*xi
            # we first figure out the Jacobian J = partial g/partial z,
            # g(z) = Az + B(z, z) + cdot * M z 
            dgdz = fcdot(t) * operators[-1] # cdot * M
            for (i, k) in enumerate(self.poly_comp):
                
                combs = list(combinations(self.einsum_ss_rhs_poly[i][1:],r=k-1))
                operands = [operators[i]] + [fz(t) for _ in range(k-1)]
                for comb in combs:
                    equation = [self.einsum_ss_rhs_poly[i][0]] + list(comb)
                    equation = ",".join(equation)

                    dgdz += np.einsum(equation,*operands)

            dxidt += dgdz.T@xi
            
            dhdzT = -fcdot(t) * operators[-2]

            for (i, k) in enumerate(self.poly_comp):
                
                combs = list(combinations(self.einsum_ss_rhs_shift_speed_numer[i][1:],r=k-1))
                operands = [operators[i + len(self.poly_comp)]] + [fz(t) for _ in range(k-1)]
                for comb in combs:
                    equation = [self.einsum_ss_rhs_shift_speed_numer[i][0]] + list(comb)
                    equation = ",".join(equation)

                    dhdzT -= np.einsum(equation,*operands)

            dxidt += (dhdzT/self.compute_shift_speed_denom(fz(t), operators)) * (np.einsum('i,i', fz(t), PhiF_dx.T@Psi@xi) + const)

        return dxidt
    
    def evaluate_shift_speed_adjoint(self,z,cdot,*operators):
        """
            evaluate (dg/dz)^T, where g = shift_speed
            where 
            dg/dz = d/dz (cdot_numer/cdot_denom)
                  = d/dz (- (p^Tz + a^TQz + ...) / (s^Tz) )
        """

        if np.linalg.norm(z) >= self.state_mag_threshold:
            print("Warning: The norm of the state vector is too large! Modified RHS of the adjoint shifting speed to zero.")
            dhdzT = 0.0*z
        else:
            
            dhdzT = -cdot * operators[-2]

            for (i, k) in enumerate(self.poly_comp):
                
                combs = list(combinations(self.einsum_ss_rhs_shift_speed_numer[i][1:],r=k-1))
                operands = [operators[i + len(self.poly_comp)]] + [z for _ in range(k-1)]
                for comb in combs:
                    equation = [self.einsum_ss_rhs_shift_speed_numer[i][0]] + list(comb)
                    equation = ",".join(equation)

                    dhdzT -= np.einsum(equation,*operands)

            dhdzT = dhdzT/self.compute_shift_speed_denom(z, operators)

        return dhdzT

    

        
        
        
        
        
        
        
        
        
