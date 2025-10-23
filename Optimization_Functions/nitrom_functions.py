import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from string import ascii_lowercase as ascii
import pymanopt

import time as tlib

def create_objective_and_gradient(manifold,opt_obj,mpi_pool,fom):
    
    """
    opt_obj:        instance of class "optimization_objects" in file "nitrom_classes.py"
    mpi_pool:       instance of the class "mpi_pool" in file "nitrom_classes.py"
    fom:            instance of the full-order model class 
    """

    euclidean_hessian = None

    @pymanopt.function.numpy(manifold)
    def cost(*params):

        """ 
            Evaluate the cost function 
            Phi and Psi:    bases (size N x r) that define the projection operator
            tensors:        (A2,A3,...)
        """
        
        Phi, Psi = params[0], params[1]
        tensors_trainable = params[2:] # A and B and p and Q
        r = Phi.shape[1]

        PhiF = Phi@sp.linalg.inv(Psi.T@Phi)
        PhiF_dx = opt_obj.take_derivative(PhiF, order=1)
        
        # Define operators derived from bases
        cdot_denom_linear = np.zeros(r)
        udx_linear = Psi.T @ PhiF_dx
        u0_dx = opt_obj.sol_template_dx
        
        for i in range(r):
            cdot_denom_linear[i] = opt_obj.inner_product(u0_dx, PhiF_dx[:, i])

        tensors = tensors_trainable + (cdot_denom_linear, udx_linear)

        J = 0.0
        J_sol = 0.0
        J_c = 0.0
        for k in range (opt_obj.my_n_traj): 

            # Integrate the reduced-order model from time t = 0 to the final time 
            # specified by the last snapshot in the training trajectory
            z_IC = Psi.T@(opt_obj.sol_fitted[k,:,0].reshape(-1))
            c_IC = opt_obj.shift_amount[k,0]
            u = Psi.T@(opt_obj.f_ext_steady[k,:].reshape(-1))
            
            output = solve_ivp(opt_obj.evaluate_rom_rhs,
                            [opt_obj.time[0],opt_obj.time[-1]],
                            np.hstack((z_IC, c_IC)),
                            method='RK45',
                            t_eval=opt_obj.time,
                            args=(u,) + tensors).y
            
            sol = output[:-1,:]
            c   = output[-1,:]
            
            e_sol = PhiF@sol - opt_obj.sol_fitted[k,:,:]
            e_c   = c - opt_obj.shift_amount[k,:]
            
            J_sol += (1./opt_obj.weight_sol[k])*np.trace(e_sol.T@e_sol)
            J_c += (1./opt_obj.weight_shift_amount[k])*np.dot(e_c,e_c)

            # print('Trajectory %d/%d: sol_error = %1.5e, c_error = %1.5e'%(k+1,opt_obj.my_n_traj,J_sol,J_c))

            J += (1./opt_obj.weight_sol[k])*np.trace(e_sol.T@e_sol) + (1./opt_obj.weight_shift_amount[k])*np.dot(e_c,e_c)
            # print('sol_error = %1.5e, c_error = %1.5e'%(np.trace(e_sol.T@e_sol),np.dot(e_c,e_c)))
            # print('sol_weight = %1.5e, c_weight = %1.5e'%(1./opt_obj.weight_sol[k],1./opt_obj.weight_shift_amount[k]))
            # print('sol_normalized_error = %1.5e, c_normalized_error = %1.5e'%((1./opt_obj.weight_sol[k])*np.trace(e_sol.T@e_sol),(1./opt_obj.weight_shift_amount[k])*np.dot(e_c,e_c)))
        
        # if opt_obj.l2_pen != None and mpi_pool.rank == 0:
        #     idx = opt_obj.poly_comp.index(1)    # index of the linear tensor
        #     time_pen = np.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom)
        #     Z = (solve_ivp(lambda t,z: tensors[idx]@z if np.linalg.norm(z) < 1e4 else 0*z,\
        #                    [0,time_pen[-1]],opt_obj.randic,method='RK45',t_eval=time_pen)).y
            
        #     J += opt_obj.l2_pen*np.dot(Z[:,-1],Z[:,-1])
            
        J = np.sum(np.asarray(mpi_pool.comm.allgather(J)))
        J_sol = np.sum(np.asarray(mpi_pool.comm.allgather(J_sol)))
        J_c = np.sum(np.asarray(mpi_pool.comm.allgather(J_c)))

        if mpi_pool.rank == 0:
            print("  Cost: %.4e = %.4e + %.4e"%(J, J_sol, J_c))

        return J

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params): 

        """ 
            Evaluate the euclidean gradient of the cost function with respect to the parameters
            Phi and Psi:    bases (size N x r) that define the projection operator
            tensors:        (A2,A3,...)
        """

        Phi, Psi = params[0], params[1]
        tensors_trainable = params[2:] # A, B, p, Q
        n, r = Phi.shape

        F = sp.linalg.inv(Psi.T@Phi)
        PhiF = Phi@F # PhiF = Phi@(Psi^T@Phi)-1
        PhiF_dx = opt_obj.take_derivative(PhiF, order=1)
        Psi_dx = opt_obj.take_derivative(Psi, order=1)

        Proj_perp = np.eye(Phi.shape[0]) - Psi@(PhiF.T)

        # Define operators derived from bases
        cdot_denom_linear = np.zeros(r)
        udx_linear = Psi.T @ PhiF_dx
        u0_dx = opt_obj.sol_template_dx
        u0_dxx = opt_obj.sol_template_dxx
        
        for i in range(r):
            cdot_denom_linear[i] = opt_obj.inner_product(u0_dx, PhiF_dx[:, i])

        tensors = tensors_trainable + (cdot_denom_linear, udx_linear)
        
        # Initialize arrays to store the gradients
        
        grad_Phi = np.zeros((n,r))
        grad_Psi = np.zeros((n,r))
        grad_tensors_trainable = [0]*len(tensors_trainable)
        
        # Initialize arrays needed for future computations
        xi_j_0 = np.zeros(r) # xi is the sum of lambda from j to N_t - 1; xi_j(t) = sum_{i = j}^{N_t - 1} lambda_i(t)
        Int_cdot_xi_z_j = np.zeros((r,r))
        Int_eta_cdot_u0dxx_outer_z_j_cdot_denom = np.zeros((n,r))
        Int_eta_cdot_z_j_outer_u0_dx_cdot_denom = np.zeros((r,n))   
        const_j = 0.0
                
        # Gauss-Legendre quadrature points and weights
        tlg, wlg = np.polynomial.legendre.leggauss(opt_obj.leggauss_deg)
        wlg = np.asarray(wlg)
        
        for k in range (opt_obj.my_n_traj): 

            # Integrate the reduced-order model from time t = 0 to the final time 
            # specified by the last snapshot in the training trajectory
            z_IC = Psi.T@(opt_obj.sol_fitted[k,:,0].reshape(-1))
            c_IC = opt_obj.shift_amount[k,0]
            u = Psi.T@(opt_obj.f_ext_steady[k,:].reshape(-1))
            
            output = solve_ivp(opt_obj.evaluate_rom_rhs,
                            [opt_obj.time[0],opt_obj.time[-1]],
                            np.hstack((z_IC, c_IC)),
                            method='RK45',
                            t_eval=opt_obj.time,
                            args=(u,) + tensors).y
            
            z   = output[:-1,:]
            sol = PhiF@z
            c   = output[-1,:]
            
            e_sol = sol - opt_obj.sol_fitted[k,:,:]
            e_c   = c - opt_obj.shift_amount[k,:]
            
            xi_j_0 *= 0.0
            Int_cdot_xi_z_j *= 0.0
            Int_eta_cdot_u0dxx_outer_z_j_cdot_denom *= 0.0
            Int_eta_cdot_z_j_outer_u0_dx_cdot_denom *= 0.0
            const_j *= 0.0
            
            for j in range(opt_obj.n_snapshots - 1):
                
                e_sol_j = e_sol[:,opt_obj.n_snapshots - j - 1] # e_sol_j is from j = N_t - 1 to j = 1
                z_j     = z[:,opt_obj.n_snapshots - j - 1]
                e_c_j   = e_c[opt_obj.n_snapshots - j - 1]
                
                grad_Phi += (2/opt_obj.weight_sol[k]) * np.einsum('i,j', e_sol_j, z_j)
                grad_Psi -= (2/opt_obj.weight_sol[k]) * np.einsum('i,j',PhiF@z_j, PhiF.T@e_sol_j)

                id1 = opt_obj.n_snapshots - j - 1 # id1 is from N_t - 1 to 1
                id0 = id1 - 1                     # id0 is from N_t - 2 to 0
                
                tf_j = opt_obj.time[id1] # from t_{N_t - 1} to t_1
                t0_j = opt_obj.time[id0] # from t_{N_t - 2} to t_0
                z_j_IC = z[:,id0]        # z_IC_j is from z_{N_t - 2} to z_0
                c_j_IC = c[id0]          # c_IC_j is from c_{N_t - 2} to c_0
                time_rom_j = np.linspace(t0_j,tf_j,num=opt_obj.nsave_rom,endpoint=True) # used to solve the adjoint eqn from t0_j to tf_j
                if np.abs(time_rom_j[-1] - tf_j) >= 1e-10:
                    print(time_rom_j[-1],tf_j)
                    raise ValueError("Error in euclidean_gradient() - final time is not correct!")
                
                output_j = solve_ivp(opt_obj.evaluate_rom_rhs,
                                [t0_j,tf_j],
                                np.hstack((z_j_IC, c_j_IC)),
                                method='RK45',
                                t_eval=time_rom_j,
                                args=(u,) + tensors).y # output_j is z and c from t0_j to tf_j with finer sampling intervals (nsave_rom points between two FOM snapshots)

                z_j_flipped = np.fliplr(output_j[:-1,:])
                fz = sp.interpolate.interp1d(time_rom_j,z_j_flipped,kind='linear',fill_value='extrapolate') # gives z(tf_j - t)
                fcdot = lambda t: opt_obj.compute_shift_speed(fz(t), tensors) # gives back cdot(z(tf_j - t))

                # Compute the adjoint ROM solution between times t0_j and tf_j
                # here xi_j = sum_{m=j}^{N_t - 1} lambda_m, and lambda_m(t_m) = (2/weight_sol)*PhiF^T e_sol_m
                # eta_j = sum_{m=j}^{N_t - 1} mu_m, mu_m(t) = z^T(t) PhiF_dx^T Psi lambda_m(t) - 2 / weight_c * e_c_m
                xi_j_0  -= (2/opt_obj.weight_sol[k])*PhiF.T@e_sol_j 
                const_j -= (2/opt_obj.weight_shift_amount[k])*e_c_j
                
                xi_j_flipped = solve_ivp(opt_obj.evaluate_rom_adjoint,
                                [t0_j,tf_j],
                                xi_j_0,
                                method='RK45',
                                t_eval=time_rom_j,
                                args=(fz, fcdot, const_j, PhiF_dx, Psi) + tensors).y

                xi_j = np.fliplr(xi_j_flipped)
                xi_j_0 = xi_j[:,0]
                z_j = np.fliplr(z_j_flipped)
                
                # Interpolate z_j and xi onto Gauss-Legendre points
                a = (tf_j - t0_j)/2
                b = (tf_j + t0_j)/2
                time_j_lg = a*tlg + b
                
                fz = sp.interpolate.interp1d(time_rom_j,z_j,kind='linear',fill_value='extrapolate')
                fxi = sp.interpolate.interp1d(time_rom_j,xi_j,kind='linear',fill_value='extrapolate')
                feta = lambda t: np.einsum('i,i', fz(t), PhiF_dx.T@Psi@fxi(t)) + const_j
                fcdot_denom = lambda t: opt_obj.compute_shift_speed_denom(fz(t), tensors)
                fcdot = lambda t: opt_obj.compute_shift_speed(fz(t), tensors)
                
                z_j_lg = fz(time_j_lg)
                xi_lg  = fxi(time_j_lg)
                eta_lg = np.array([feta(t) for t in time_j_lg])
                cdot_denom_lg = np.array([fcdot_denom(t) for t in time_j_lg])
                cdot_lg = np.array([fcdot(t) for t in time_j_lg])
                
                for i in range (opt_obj.leggauss_deg):

                    Int_cdot_xi_z_j += a*wlg[i]*cdot_lg[i] * np.einsum('i, j', xi_lg[:,i], z_j_lg[:,i])
                    Int_eta_cdot_u0dxx_outer_z_j_cdot_denom += a*wlg[i]*eta_lg[i]*cdot_lg[i] * opt_obj.outer_product(u0_dxx, z_j_lg[:,i]) / cdot_denom_lg[i]
                    Int_eta_cdot_z_j_outer_u0_dx_cdot_denom += a*wlg[i]*eta_lg[i]*cdot_lg[i] * opt_obj.outer_product(z_j_lg[:,i], u0_dx) / cdot_denom_lg[i]
                    grad_Phi += Psi_dx @ Int_cdot_xi_z_j - Int_eta_cdot_u0dxx_outer_z_j_cdot_denom

                    grad_Psi -= PhiF_dx @ Int_cdot_xi_z_j.T
                    grad_Psi += PhiF @ (Int_cdot_xi_z_j.T @ Psi.T) @ PhiF_dx
                    grad_Psi -= PhiF @ Int_eta_cdot_z_j_outer_u0_dx_cdot_denom @ PhiF_dx

                    for (count,p) in enumerate(opt_obj.poly_comp):
                        equation = ','.join(ascii[:p+1])
                        operands = [xi_lg[:,i]] + [z_j_lg[:,i] for _ in range (p)]
                        grad_tensors_trainable[count] -= a*wlg[i]*np.einsum(equation,*operands)
                        equation = ','.join(ascii[:p])
                        operands = [z_j_lg[:,i] for _ in range (p)]
                        grad_tensors_trainable[count + len(opt_obj.poly_comp)] += a*wlg[i]*np.einsum(equation,*operands) * eta_lg[i] / cdot_denom_lg[i]
                        
            e_sol_0 = e_sol[:,0]
            z_0     = z[:,0]
            xi_j_0  -= (2/opt_obj.weight_sol[k])*PhiF.T@e_sol_0 
            grad_Phi += (2/opt_obj.weight_sol[k])*np.einsum('i,j', e_sol_0, z_0)
            grad_Phi = (Proj_perp @ grad_Phi) @ F.T
            grad_Psi -= (2/opt_obj.weight_sol[k])*np.einsum('i,j', PhiF@z_0, PhiF.T@e_sol_0) - np.einsum('i,j', opt_obj.sol_fitted[k,:,0], xi_j_0)
               
        """ The original codes in NiTROM      
        # for k in range (opt_obj.my_n_traj):

        #     z0 = Psi.T@opt_obj.X[k,:,0]
        #     u = Psi.T@opt_obj.F[:,k]
        #     sol = solve_ivp(opt_obj.evaluate_rom_rhs,[0,opt_obj.time[-1]],z0,\
        #                     method='RK45',t_eval=opt_obj.time,args=(u,) + tensors)
        #     Z = sol.y
        #     e = fom.compute_output(opt_obj.X[k,:,:]) - fom.compute_output(PhiF@Z)
        #     alpha = opt_obj.weights[k]
            
        #     lam_j_0 *= 0.0
        #     Int_lambda *= 0.0
            
        #     for j in range (opt_obj.n_snapshots - 1):
        
        #         ej = e[:,opt_obj.n_snapshots - j - 1]
        #         zj = Z[:,opt_obj.n_snapshots - j - 1]
        #         Ctej = fom.compute_output_derivative(PhiF@zj).T@ej

        #         # Compute the sums in (2.13) and (2.14) in the arXiv paper. Notice that this loop sums backwards
        #         # from j = N-1 to j = 1, so we will compute the term j = 0 after this loop 
        #         grad_Psi += (2/alpha)*np.einsum('i,j',PhiF@zj,PhiF.T@Ctej)
        #         grad_Phi += -(2/alpha)*np.einsum('i,j',Ctej - Psi@(PhiF.T@Ctej),F@zj)


        #         # ------ Compute the fwd ROM solution between times t0_j and tf_j ---------
        #         id1 = opt_obj.n_snapshots - 1 - j
        #         id0 = id1 - 1
                
        #         tf_j = opt_obj.time[id1]
        #         t0_j = opt_obj.time[id0]
        #         z0_j = Z[:,id0]
                
        #         time_rom_j = np.linspace(t0_j,tf_j,num=opt_obj.nsave_rom,endpoint=True)
        #         if np.abs(time_rom_j[-1] - tf_j) >= 1e-10:
        #             print(time_rom_j[-1],tf_j)
        #             raise ValueError("Error in euclidean_gradient() - final time is not correct!")
                
        #         sol_j = solve_ivp(opt_obj.evaluate_rom_rhs,[t0_j,tf_j],z0_j,method='RK45',\
        #                           t_eval=time_rom_j,args=(u,) + tensors)
        #         Z_j = np.fliplr(sol_j.y)
        #         fZ = sp.interpolate.interp1d(time_rom_j,Z_j,kind='linear',fill_value='extrapolate')
        #         # --------------------------------------------------------------------------

        #         # ------ Compute the adj ROM solution between times t0_j and tf_j ----------
        #         lam_j_0 += (2/alpha)*PhiF.T@Ctej
        #         sol_lam = solve_ivp(opt_obj.evaluate_rom_adjoint,[t0_j,tf_j],lam_j_0,\
        #                             method='RK45',t_eval=time_rom_j,args=(fZ,) + tensors)
        #         Lam = np.fliplr(sol_lam.y)
        #         lam_j_0 = Lam[:,0]
        #         Z_j = np.fliplr(Z_j)
        #         # --------------------------------------------------------------------------
                
        #         # Interpolate Z_j and Lam onto Gauss-Legendre points
        #         a = (tf_j - t0_j)/2
        #         b = (tf_j + t0_j)/2
        #         time_j_lg = a*tlg + b

        #         fZ = sp.interpolate.interp1d(time_rom_j,Z_j,kind='linear',fill_value='extrapolate')
        #         fL = sp.interpolate.interp1d(time_rom_j,Lam,kind='linear',fill_value='extrapolate')
        #         Z_j_lg = fZ(time_j_lg)
        #         Lam_lg = fL(time_j_lg)
                
        #         for i in range (opt_obj.leggauss_deg):
                    
        #             Int_lambda += a*wlg[i]*Lam_lg[:,i]
                    
        #             for (count,p) in enumerate(opt_obj.poly_comp):
        #                 equation = ','.join(ascii[:p+1])
        #                 operands = [Lam_lg[:,i]] + [Z_j_lg[:,i] for _ in range (p)]
        #                 grad_tensors[count] -= a*wlg[i]*np.einsum(equation,*operands)
                    
            
            # Add the term j = 0 in the sums (2.13) and (2.14). Also add the  
            # contribution of the initial condition (last term in (2.14)) to grad_Psi.
            # Add also the contribution of the steady forcing to grad_Psi.
            ej, zj = e[:,0], Z[:,0]
            Ctej = fom.compute_output_derivative(PhiF@zj).T@ej
            grad_Psi += (2/alpha)*np.einsum('i,j',PhiF@zj,PhiF.T@Ctej) \
                        - np.einsum('i,j',opt_obj.X[k,:,0],lam_j_0) \
                        - np.einsum('i,j',opt_obj.F[:,k],Int_lambda)
            grad_Phi += -(2/alpha)*np.einsum('i,j',Ctej - Psi@(PhiF.T@Ctej),F@zj)
        """

        
        # # Compute the gradient of the stability-promoting term
        # if opt_obj.l2_pen != None and mpi_pool.rank == 0:
            
        #     idx = opt_obj.poly_comp.index(1)    # index of the linear tensor
            
        #     A = tensors[idx]
            
        #     time_pen = np.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom)
        #     Z = (solve_ivp(lambda t,z: A@z if np.linalg.norm(z) < 1e4 else 0*z,\
        #                    [0,time_pen[-1]],opt_obj.randic,method='RK45',t_eval=time_pen)).y
        #     Mu = (solve_ivp(lambda t,z: A.T@z if np.linalg.norm(z) < 1e4 else 0*z,\
        #                    [0,time_pen[-1]],-2*opt_obj.l2_pen*Z[:,-1],method='RK45',t_eval=time_pen)).y
        #     Mu = np.fliplr(Mu)
            
            
        #     for k in range (opt_obj.n_snapshots - 1):
                
        #         k0, k1 = k*opt_obj.nsave_rom, (k+1)*opt_obj.nsave_rom
        #         fZ = sp.interpolate.interp1d(time_pen[k0:k1],Z[:,k0:k1],kind='linear',fill_value='extrapolate')
        #         fMu = sp.interpolate.interp1d(time_pen[k0:k1],Mu[:,k0:k1],kind='linear',fill_value='extrapolate')
            
        #         a = (time_pen[k1-1] - time_pen[k0])/2
        #         b = (time_pen[k1-1] + time_pen[k0])/2
        #         time_k_lg = a*tlg + b
                
        #         Zk = fZ(time_k_lg)
        #         Muk = fMu(time_k_lg)
                
        #         for i in range (opt_obj.leggauss_deg):
        #             grad_tensors[idx] += -a*wlg[i]*np.einsum('i,j',Muk[:,i],Zk[:,i])
                
        if opt_obj.which_fix == 'fix_bases':

            grad_Phi *= 0.0; grad_Psi *= 0.0
            for k in range (len(grad_tensors_trainable)):
                grad_tensors_trainable[k] = sum(mpi_pool.comm.allgather(grad_tensors_trainable[k]))


        elif opt_obj.which_fix == 'fix_tensors':    
            
            for k in range (len(grad_tensors_trainable)): grad_tensors_trainable[k] *= 0.0
            grad_Phi = sum(mpi_pool.comm.allgather(grad_Phi))
            grad_Psi = sum(mpi_pool.comm.allgather(grad_Psi))


        else: 

            grad_Phi = sum(mpi_pool.comm.allgather(grad_Phi))
            grad_Psi = sum(mpi_pool.comm.allgather(grad_Psi))
            for k in range (len(grad_tensors_trainable)):
                grad_tensors_trainable[k] = sum(mpi_pool.comm.allgather(grad_tensors_trainable[k]))

        return grad_Phi, grad_Psi, *grad_tensors_trainable


    return cost, euclidean_gradient, euclidean_hessian

"""
Deprecated: check gradient using self-programmed finite difference

def check_gradient_using_finite_difference(M,Phi,Psi, A, B, p, Q, opt_obj,mpi_pool,fom,eps):

    cost, grad, _ = create_objective_and_gradient(M,opt_obj,mpi_pool,fom)
    gPhi, gPsi, gA, gB, gp, gQ = grad(Phi,Psi, A, B, p, Q)

    # Check Phi gradient 
    delta = sp.linalg.orth(np.random.randn(3,2))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi + eps*delta,Psi,A2,A3,A4) - cost(Phi - eps*delta,Psi,A2,A3,A4))
    dgrad = np.trace(delta.T@gPhi)
    error = np.abs(dfd - dgrad)
    percent_error = error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for Phi ------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")


    # Check Psi gradient 
    delta = sp.linalg.orth(np.random.randn(3,2))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi,Psi + eps*delta,A2,A3,A4) - cost(Phi,Psi - eps*delta,A2,A3,A4))
    dgrad = np.trace(delta.T@gPsi)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for Psi ------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")


    # Check A2 gradient 
    delta = np.random.randn(2,2)
    delta = delta/np.sqrt(np.trace(delta.T@delta))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi,Psi,A2 + eps*delta,A3,A4) - cost(Phi,Psi,A2 - eps*delta,A3,A4))
    dgrad = np.trace(delta.T@gA2)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for A2 -------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")

    # Check A3 gradient 
    delta = np.random.randn(2,2,2)
    delta = delta/np.sqrt(np.einsum('ijk,ijk',delta,delta))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi,Psi,A2,A3 + eps*delta,A4) - cost(Phi,Psi,A2,A3 - eps*delta,A4))
    dgrad = np.einsum('ijk,ijk',delta,gA3)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for A3 -------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")


    # Check A4 gradient 
    delta = np.random.randn(2,2,2,2)
    delta = delta/np.sqrt(np.einsum('ijkl,ijkl',delta,delta))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)
        
    dfd = (0.5/eps)*(cost(Phi,Psi,A2,A3,A4 + eps*delta) - cost(Phi,Psi,A2,A3,A4 - eps*delta))
    dgrad = np.einsum('ijkl,ijkl',delta,gA4)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for A4 -------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")
"""




