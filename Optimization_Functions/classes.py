import numpy as np 
from mpi4py import MPI
from itertools import combinations
from string import ascii_lowercase as ascii

class mpi_pool:

    def __init__(self,comm,n_sol,**kwargs):
        
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

        self.n_sol = n_sol                        # Total number of training trajectories
        if self.size > self.n_sol:
            raise ValueError ("You have more MPI processes than trajectories!")
        else:
            if self.rank == 0:
                print("Hello, you are running NiTROM with %d MPI processors."%self.size)

        self.my_n_sol = self.n_sol//self.size     # Number of trajectories owned by process self.rank
        self.my_n_sol += 1 if np.mod(self.n_sol,self.size) > self.rank else 0

        # Vectors used for future MPI communications
        self.counts = np.zeros(self.size,dtype=np.int64)    
        self.comm.Allgather([np.asarray([self.my_n_sol]),MPI.INT],[self.counts,MPI.INT])
        self.disps = np.concatenate(([0],np.cumsum(self.counts)[:-1]))
        
        self.kwargs = kwargs
        # Load data from file
        
    def load_data(self):
        self.time = np.load(self.kwargs.get('fname_time', None))
        self.load_template(self.kwargs)
        self.load_sol(self.kwargs)
        self.load_rhs(self.kwargs)
        self.load_shift(self.kwargs)
        self.load_weight(self.kwargs)
        self.load_steady_forcing(self.kwargs)
        
    def load_template(self,kwargs):
        
        fname_sol_template = kwargs.get('fname_sol_template',None)
        if fname_sol_template == None:
            raise ValueError ("Need a valid fname_sol_template path where you store your solution template")
        else:
            self.sol_template = np.load(fname_sol_template)
            
        fname_sol_template_dx = kwargs.get('fname_sol_template_dx',None)
        if fname_sol_template_dx == None:
            raise ValueError ("Need a valid fname_sol_template_dx path where you store your solution template derivative")
        else:
            self.sol_template_dx = np.load(fname_sol_template_dx)
            
        fname_sol_template_dxx = kwargs.get('fname_sol_template_dxx',None)
        if fname_sol_template_dxx == None:
            raise ValueError ("Need a valid fname_sol_template_dxx path where you store your solution template second derivative")
        else:
            self.sol_template_dxx = np.load(fname_sol_template_dxx)

    def load_sol(self,kwargs):
        
        fname_sol = kwargs.get('fname_sol',None)
        if fname_sol == None:
            raise ValueError ("Need a valid fname_sol path where you store your solutions")
        else:
            self.fnames_sol = [fname_sol%(k+self.disps[self.rank]) for k in range (self.my_n_sol)]
            X = [np.load(self.fnames_sol[k]) for k in range (self.my_n_sol)]
            self.N, self.n_snapshots = X[0].shape
            
            self.sol = np.zeros((self.my_n_sol,self.N,self.n_snapshots))
            
            for k in range (self.my_n_sol): self.sol[k,] = X[k]
            fname_sol_fitted = kwargs.get('fname_sol_fitted',None)
            if fname_sol_fitted == None:
                raise ValueError ("Need a valid fname_sol_fitted path where you store your template fitted solutions")
            else:
                self.fnames_sol_fitted = [fname_sol_fitted%(k+self.disps[self.rank]) for k in range (self.my_n_sol)]
                X_fitted = [np.load(self.fnames_sol_fitted[k]) for k in range (self.my_n_sol)]
                
                self.sol_fitted = np.zeros((self.my_n_sol,self.N,self.n_snapshots))
        
                for k in range (self.my_n_sol): self.sol_fitted[k,] = X_fitted[k]

            fname_sol_init = kwargs.get('fname_sol_init',None)
            if fname_sol_init == None:
                raise ValueError ("Need a valid fname_sol_init path where you store your initial solutions")
            else:
                self.fnames_sol_init = [fname_sol_init%(k+self.disps[self.rank]) for k in range (self.my_n_sol)]
                X_init = [np.load(self.fnames_sol_init[k]) for k in range (self.my_n_sol)]
                
                self.sol_init = np.zeros((self.my_n_sol,self.N))

                for k in range (self.my_n_sol): self.sol_init[k,] = X_init[k]

            fname_sol_init_fitted = kwargs.get('fname_sol_init_fitted',None)
            if fname_sol_init_fitted == None:
                raise ValueError ("Need a valid fname_sol_init_fitted path where you store your initial fitted solutions")
            else:
                self.fnames_sol_init_fitted = [fname_sol_init_fitted%(k+self.disps[self.rank]) for k in range (self.my_n_sol)]
                X_init_fitted = [np.load(self.fnames_sol_init_fitted[k]) for k in range (self.my_n_sol)]

                self.sol_init_fitted = np.zeros((self.my_n_sol,self.N))

                for k in range (self.my_n_sol): self.sol_init_fitted[k,] = X_init_fitted[k]

    def load_rhs(self,kwargs):
        
        fname_rhs = kwargs.get('fname_rhs',None)
        if fname_rhs != None:
            self.fnames_rhs = [fname_rhs%(k+self.disps[self.rank]) for k in range (self.my_n_sol)]
            dXdt = [np.load(self.fnames_rhs[k]) for k in range (self.my_n_sol)]
            
            self.rhs = np.zeros((self.my_n_sol,self.N,self.n_snapshots))
            
            for k in range (self.my_n_sol): self.rhs[k,] = dXdt[k]
        fname_rhs_fitted = kwargs.get('fname_rhs_fitted',None)
        if fname_rhs_fitted != None:
            self.fnames_rhs_fitted = [fname_rhs_fitted%(k+self.disps[self.rank]) for k in range (self.my_n_sol)]
            dXdt_fitted = [np.load(self.fnames_rhs_fitted[k]) for k in range (self.my_n_sol)]
            
            self.rhs_fitted = np.zeros((self.my_n_sol,self.N,self.n_snapshots))
            
            for k in range (self.my_n_sol): self.rhs_fitted[k,] = dXdt_fitted[k]
            
    def load_shift(self,kwargs):
        fname_shift_amount = kwargs.get('fname_shift_amount',None)
        if fname_shift_amount != None:
            self.fnames_shift_amount = [fname_shift_amount%(k+self.disps[self.rank]) for k in range (self.my_n_sol)]
            c = [np.load(self.fnames_shift_amount[k]) for k in range (self.my_n_sol)]

            self.shift_amount = np.zeros((self.my_n_sol, self.n_snapshots))

            for k in range (self.my_n_sol): self.shift_amount[k,] = c[k]

        fname_shift_speed = kwargs.get('fname_shift_speed',None)
        if fname_shift_speed != None:
            self.fnames_shift_speed = [fname_shift_speed%(k+self.disps[self.rank]) for k in range (self.my_n_sol)]
            cdot = [np.load(self.fnames_shift_speed[k]) for k in range (self.my_n_sol)]

            self.shift_speed = np.zeros((self.my_n_sol, self.n_snapshots))

            for k in range (self.my_n_sol): self.shift_speed[k,] = cdot[k]

    def load_weight(self,kwargs):

        fname_weight_sol = kwargs.get('fname_weight_sol',None)
        self.weight_sol = np.ones(self.my_n_sol)
        if fname_weight_sol != None:
            self.fnames_weight_sol = [fname_weight_sol%(k+self.disps[self.rank]) for k in range (self.my_n_sol)]
            alpha = [np.load(self.fnames_weight_sol[k]) for k in range (self.my_n_sol)]
            self.weight_sol = np.zeros(self.my_n_sol)
            for k in range (self.my_n_sol): self.weight_sol[k] = alpha[k]
            
        fname_weight_shift_amount = kwargs.get('fname_weight_shift_amount',None)
        self.weight_shift_amount = np.ones(self.my_n_sol)
        if fname_weight_shift_amount != None:
            self.fnames_weight_shift_amount = [fname_weight_shift_amount%(k+self.disps[self.rank]) for k in range (self.my_n_sol)]
            beta = [np.load(self.fnames_weight_shift_amount[k]) for k in range (self.my_n_sol)]
            self.weight_shift_amount = np.zeros(self.my_n_sol)
            for k in range (self.my_n_sol): self.weight_shift_amount[k] = beta[k]
            
    def load_steady_forcing(self,kwargs):
        
        fname_forcing = kwargs.get('fname_steady_forcing',None)
        self.f_ext_steady = np.zeros((self.my_n_sol, self.N))
        if fname_forcing != None:
            self.fnames_forcing = [(fname_forcing)%(k+self.disps[self.rank]) for k in range (self.my_n_sol)]
            for k in range (self.my_n_sol):  self.f_ext_steady[k,:] = np.load(self.fnames_forcing[k])

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
        
        # self.sol_init = mpi_pool.sol_init[which_trajs,:]
        # self.sol_init_fitted = mpi_pool.sol_init_fitted[which_trajs,:]
        self.sol = mpi_pool.sol[which_trajs,:,:]
        self.sol_fitted = mpi_pool.sol_fitted[which_trajs,:,:]
        self.rhs = mpi_pool.rhs[which_trajs,:,:]
        self.rhs_fitted = mpi_pool.rhs_fitted[which_trajs,:,:]
        self.shift_amount = mpi_pool.shift_amount[which_trajs,:]
        self.shift_speed = mpi_pool.shift_speed[which_trajs,:]
        self.f_ext_steady = mpi_pool.f_ext_steady[which_trajs,:]
        self.weight_sol = mpi_pool.weight_sol[which_trajs]
        self.weight_shift_amount = mpi_pool.weight_shift_amount[which_trajs]

        self.sol = self.sol[:,:,which_times]
        self.sol_fitted = self.sol_fitted[:,:,which_times]
        self.rhs = self.rhs[:,:,which_times]
        self.rhs_fitted = self.rhs_fitted[:,:,which_times]
        self.shift_amount = self.shift_amount[:,which_times]
        self.shift_speed = self.shift_speed[:,which_times]
        self.time = mpi_pool.time[which_times]
        
        self.my_n_traj, _, self.n_snapshots = self.sol.shape
        self.leggauss_deg = leggauss_deg
        self.nsave_rom = nsave_rom
        self.poly_comp = poly_comp
        
        self.generate_einsum_subscripts_rhs_poly()
        self.generate_einsum_subscripts_rhs_shift_speed_numer()
        
        # Count the total number of trajectories in this batch and
        # scale the weight accordingly so that the cost function measures
        # the average error over snapshots and trajectories. (Notice that 
        # if all trajectories are loaded, then np.sum(counts) = mpi_pool.n_traj)
        counts = np.zeros(mpi_pool.size,dtype=np.int64)
        mpi_pool.comm.Allgather([np.asarray([self.my_n_traj]),MPI.INT],[counts,MPI.INT])
        
        # Parse the keyword arguments
        self.relative_weight = kwargs.get('relative_weight',1.0)
        
        for idx in range(self.my_n_traj):
            self.weight_sol[idx] = np.mean(np.linalg.norm(self.sol[idx,:,:],axis=0)**2)
            self.weight_shift_amount[idx] = np.mean((self.shift_amount[idx,:] - self.shift_amount[idx,0])**2)        
        
        self.weight_sol *= np.sum(counts)*self.n_snapshots
        self.weight_shift_amount *= np.sum(counts)*self.n_snapshots / self.relative_weight
        
        self.sol_template_dx = kwargs.get('sol_template_dx',None)
        self.sol_template_dxx = kwargs.get('sol_template_dxx',None)
        self.take_derivative = kwargs.get('spatial_derivative_method',None)
        self.inner_product = kwargs.get('inner_product_method',None)
        self.outer_product = kwargs.get('outer_product_method',None)
        
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

        if np.linalg.norm(z) >= 1e4:
            dzdt = 0.0*z
            dcdt = 0.0
            raise ValueError ("The norm of the state vector is too large!")
        else:
            f = kwargs.get('forcing_interp',None)
            f = f(t) if f != None else np.zeros(len(z))
            u = u.copy() if hasattr(u,"__len__") == True else u(t)
            dzdt = u + f
            
            cdot_denom_linear = operators[-2]
            udx_linear = operators[-1]
            cdot_denom = np.einsum('i,i',cdot_denom_linear,z)
            # print("cdot_denom:", cdot_denom)
            if abs(cdot_denom) < 1e-4:
                raise ValueError ("Denominator in reconstruction equation of the shifting speed is too close to zero!")
                # print("Denominator in reconstruction equation of the shifting speed is too close to zero!")
                # cdot_denom = 1e-2 * np.sign(cdot_denom)
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

            # if abs(dcdt) > 1e4:
            #     raise ValueError ("The shift speed is too large!")
                
        return np.hstack((dzdt, dcdt))

    def compute_shift_speed(self, z, operators):
        """
            Function to compute the shift speed given a state z and the ROM operators
        """
        cdot_denom_linear = operators[-2]
        cdot_denom = np.einsum('i,i',cdot_denom_linear,z)
        if abs(cdot_denom) < 1e-4:
            raise ValueError ("Denominator in reconstruction equation of the shifting speed is too close to zero!")
        
        cdot_numer = 0.0
        
        for (i, k) in enumerate(self.poly_comp):
            equation = ",".join(self.einsum_ss_rhs_shift_speed_numer[i])
            operands = [operators[i + len(self.poly_comp)]] + [z for _ in range(k)]
            cdot_numer -= np.einsum(equation,*operands)

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
        if abs(cdot_denom) < 1e-4:
            raise ValueError ("Denominator in reconstruction equation of the shifting speed is too close to zero!")
        
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

        if np.linalg.norm(xi) >= 1e4:
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

    

        
        
        
        
        
        
        
        
        
