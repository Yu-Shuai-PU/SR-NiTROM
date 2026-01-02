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
            (weighted with the square root of the inner product matrix such that all inner products are standard Euclidean inner products)
            tensors:        (A2,A3,...)
        """
        
        Phi, Psi = params[0], params[1]
        tensors_trainable = params[2:] # A and B and p and Q
        r = Phi.shape[1]

        PhiF = Phi@sp.linalg.inv(Psi.T@Phi)
        PhiF_dx = opt_obj.spatial_deriv(PhiF, order=1)
        
        # Define operators derived from bases
        cdot_denom_linear = np.zeros(r)
        udx_linear = Psi.T @ PhiF_dx
        u0_dx_weighted = opt_obj.X_template_dx_weighted
        
        for i in range(r):
            cdot_denom_linear[i] = np.dot(u0_dx_weighted, PhiF_dx[:, i])

        tensors = tensors_trainable + (cdot_denom_linear, udx_linear)

        J = 0.0
        J_X = 0.0
        J_c = 0.0
        J_cdot = 0.0
        for k in range (opt_obj.my_n_traj): 

            # Integrate the reduced-order model from time t = 0 to the final time 
            # specified by the last snapshot in the training trajectory
            z0 = Psi.T@(opt_obj.X_fitted_weighted[k,:,0].reshape(-1))
            u = Psi.T@(opt_obj.F_weighted[:,k].reshape(-1))
            c0 = opt_obj.c[k,0]
            sol = solve_ivp(opt_obj.evaluate_rom_rhs,
                            [opt_obj.time[0],opt_obj.time[-1]],
                            np.hstack((z0, c0)),
                            method='RK45',
                            t_eval=opt_obj.time,
                            args=(u,) + tensors).y
            sol_z = sol[:-1,:]
            sol_c = sol[-1,:]
            sol_cdot = np.array([opt_obj.compute_shift_speed(sol_z[:,i], tensors) for i in range (opt_obj.n_snapshots)])
            sol_X_fitted_weighted = PhiF@sol_z
            
            sol_X_weighted = np.zeros_like(sol_X_fitted_weighted)
            for j in range (opt_obj.n_snapshots):
                sol_X_weighted[:,j] = opt_obj.shift(sol_X_fitted_weighted[:,j], sol_c[j])
            
            e_X_weighted = opt_obj.X_weighted[k,:,:] - sol_X_weighted
            e_c = opt_obj.c[k,:] - sol_c
            e_cdot = opt_obj.cdot[k,:] - sol_cdot
            
            error_X = np.sum(np.sum(e_X_weighted**2,axis=0) * opt_obj.weights_X[k,:])
            error_c = np.sum(e_c**2 * opt_obj.weights_c[k,:])
            error_cdot = np.sum(e_cdot**2 * opt_obj.weights_cdot[k,:])
           
            J_X += error_X
            J_c   += error_c
            J_cdot += error_cdot
            J += error_X + error_c + error_cdot

        if opt_obj.l2_pen != None and mpi_pool.rank == 0:
            idx = opt_obj.poly_comp.index(1)    # index of the linear tensor
            time_pen = np.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom)
            Z = (solve_ivp(lambda t,z: tensors[idx]@z if np.linalg.norm(z) < 1e4 else 0*z,\
                           [0,time_pen[-1]],opt_obj.randic,method='RK45',t_eval=time_pen)).y
            
            J += opt_obj.l2_pen*np.dot(Z[:,-1],Z[:,-1])
            
        J = np.sum(np.asarray(mpi_pool.comm.allgather(J)))
        J_X = np.sum(np.asarray(mpi_pool.comm.allgather(J_X)))
        J_c = np.sum(np.asarray(mpi_pool.comm.allgather(J_c)))
        J_cdot = np.sum(np.asarray(mpi_pool.comm.allgather(J_cdot)))

        if mpi_pool.rank == 0:
            print("  Cost: %.4e = %.4e + %.4e + %.4e"%(J, J_X, J_c, J_cdot))

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
        
        # Initialize arrays to store the gradients
        n, r = Phi.shape
        grad_Phi = np.zeros((n,r))
        grad_Psi = np.zeros((n,r))
        grad_tensors_trainable = [0]*len(tensors_trainable)
        
        # Initialize arrays needed for future computations
        xi0_j = np.zeros(r) # xi is the sum of lambda ("lam" in the original NiTROM codes) from j to N_t - 1; xi_j(t) = sum_{i = j}^{N_t - 1} lambda_i(t)
        Int_cdot_xi_z_j = np.zeros((r,r))
        Int_eta_cdot_u0dxx_outer_z_j_cdot_denom = np.zeros((n,r))
        const_j = 0.0
        
        # Biorthogonalize Phi and Psi
        F = sp.linalg.inv(Psi.T@Phi)
        PhiF = Phi@F # PhiF = Phi@(Psi^T@Phi)-1
        PhiF_dx = opt_obj.spatial_deriv(PhiF, order=1)
        Psi_dx = opt_obj.spatial_deriv(Psi, order=1)
        
        # Gauss-Legendre quadrature points and weights
        tlg, wlg = np.polynomial.legendre.leggauss(opt_obj.leggauss_deg)
        wlg = np.asarray(wlg)
        
        cdot_denom_linear = np.zeros(r)
        udx_linear = Psi.T @ PhiF_dx
        u0_dx_weighted = opt_obj.X_template_dx_weighted
        u0_dxx_weighted = opt_obj.X_template_dxx_weighted
        
        for i in range(r):
            cdot_denom_linear[i] = np.dot(u0_dx_weighted, PhiF_dx[:, i])

        tensors = tensors_trainable + (cdot_denom_linear, udx_linear)

        for k in range (opt_obj.my_n_traj): 

            # Integrate the reduced-order model from time t = 0 to the final time 
            # specified by the last snapshot in the training trajectory
            z0 = Psi.T@(opt_obj.X_fitted_weighted[k,:,0].reshape(-1))
            c0 = opt_obj.c[k,0]
            u = Psi.T@(opt_obj.F_weighted[:,k].reshape(-1))
            
            sol = solve_ivp(opt_obj.evaluate_rom_rhs,
                            [opt_obj.time[0],opt_obj.time[-1]],
                            np.hstack((z0, c0)),
                            method='RK45',
                            t_eval=opt_obj.time,
                            args=(u,) + tensors).y
            
            Z = sol[:-1,:]
            c = sol[-1,:]
            cdot = np.array([opt_obj.compute_shift_speed(Z[:,i], tensors) for i in range (opt_obj.n_snapshots)])
            
            sol_X_fitted_weighted = PhiF@Z
            X_weighted_fitted_with_c = np.zeros_like(sol_X_fitted_weighted)
            for j in range (opt_obj.n_snapshots):
                X_weighted_fitted_with_c[:,j] = opt_obj.shift(opt_obj.X_weighted[k,:,j], -c[j])
                
            e_X_fitted_weighted = X_weighted_fitted_with_c - sol_X_fitted_weighted # recall that ||(S_{c_ROM}[u_hat] - u)||_2 = ||(u_hat - S_{-c_ROM}[u])||_2
            e_c = opt_obj.c[k,:] - c
            e_cdot = opt_obj.cdot[k,:] - cdot
            
            xi0_j *= 0.0
            Int_cdot_xi_z_j *= 0.0
            Int_eta_cdot_u0dxx_outer_z_j_cdot_denom *= 0.0
            const_j *= 0.0
            
            for j in range(opt_obj.n_snapshots - 1):
                
                e_X_fitted_weighted_j = e_X_fitted_weighted[:,opt_obj.n_snapshots - j - 1] # e_sol_j is from j = N_t - 1 to j = 1
                z_j     = Z[:,opt_obj.n_snapshots - j - 1]
                e_c_j    = e_c[opt_obj.n_snapshots - j - 1]
                e_cdot_j = e_cdot[opt_obj.n_snapshots - j - 1]
                cdot_j = cdot[opt_obj.n_snapshots - j - 1]
                cdot_denom_j = opt_obj.compute_shift_speed_denom(z_j, tensors)
                u0dxx_zj_outer_product = np.outer(u0_dxx_weighted, z_j)
                
                weight_X_j = opt_obj.weights_X[k,opt_obj.n_snapshots - j - 1]
                weight_c_j = opt_obj.weights_c[k,opt_obj.n_snapshots - j - 1]
                weight_cdot_j = opt_obj.weights_cdot[k,opt_obj.n_snapshots - j - 1]
                
                grad_Phi -= 2 * weight_X_j * np.einsum('i,j', e_X_fitted_weighted_j, z_j)
                grad_Phi -= 2 * weight_cdot_j * e_cdot_j * cdot_j * u0dxx_zj_outer_product / cdot_denom_j
                grad_Psi += 2 * weight_X_j * np.einsum('i,j',PhiF@z_j, PhiF.T@e_X_fitted_weighted_j)
                grad_Psi += 2 * weight_cdot_j * e_cdot_j * cdot_j * PhiF @ (u0dxx_zj_outer_product.T @ PhiF) / cdot_denom_j ### Warning: check this term carefully!!! Heavy memory usage!!!
                
                for (count,p) in enumerate(opt_obj.poly_comp):
                    equation = ','.join(ascii[:p])
                    operands = [z_j for _ in range (p)]
                    grad_tensors_trainable[count + len(opt_obj.poly_comp)] += 2 * weight_cdot_j * e_cdot_j * np.einsum(equation,*operands) / cdot_denom_j 

                id1 = opt_obj.n_snapshots - j - 1 # id1 is from N_t - 1 to 1
                id0 = id1 - 1                     # id0 is from N_t - 2 to 0
                
                tf_j = opt_obj.time[id1] # from t_{N_t - 1} to t_1
                t0_j = opt_obj.time[id0] # from t_{N_t - 2} to t_0
                z0_j = Z[:,id0]        # z_IC_j is from z_{N_t - 2} to z_0
                c0_j = c[id0]          # c_IC_j is from c_{N_t - 2} to c_0
                
                time_rom_j = np.linspace(t0_j,tf_j,num=opt_obj.nsave_rom,endpoint=True) # used to solve the adjoint eqn from t0_j to tf_j
                if np.abs(time_rom_j[-1] - tf_j) >= 1e-10:
                    print(time_rom_j[-1],tf_j)
                    raise ValueError("Error in euclidean_gradient() - final time is not correct!")
                
                sol_j = solve_ivp(opt_obj.evaluate_rom_rhs,
                                [t0_j,tf_j],
                                np.hstack((z0_j, c0_j)),
                                method='RK45',
                                t_eval=time_rom_j,
                                args=(u,) + tensors).y # output_j is z and c from t0_j to tf_j with finer sampling intervals (nsave_rom points between two FOM snapshots)

                sol_z_j_flipped = np.fliplr(sol_j[:-1,:])
                fz = sp.interpolate.interp1d(time_rom_j,sol_z_j_flipped,kind='linear',fill_value='extrapolate') # gives z(tf_j - t)
                fcdot = lambda t: opt_obj.compute_shift_speed(fz(t), tensors) # gives back cdot(z(tf_j - t))

                # Compute the adjoint ROM solution between times t0_j and tf_j
                # here xi_j = sum_{m=j}^{N_t - 1} lambda_m, and lambda_m(t_m) = (2/weight_sol)*PhiF^T e_sol_m
                # eta_j = sum_{m=j}^{N_t - 1} mu_m, mu_m(t) = z^T(t) PhiF_dx^T Psi lambda_m(t) - 2 / weight_c * e_c_m
                xi0_j  += 2 * weight_X_j * PhiF.T@e_X_fitted_weighted_j 
                xi0_j  += 2 * weight_cdot_j * e_cdot_j * opt_obj.evaluate_shift_speed_adjoint(z_j, cdot_j, *tensors)
                const_j -= 2 * weight_X_j * np.einsum("i,i", e_X_fitted_weighted_j, PhiF_dx @ z_j)
                const_j += 2 * weight_c_j * e_c_j
               
                sol_xi_j_flipped = solve_ivp(opt_obj.evaluate_rom_adjoint,
                                [t0_j,tf_j],
                                xi0_j,
                                method='RK45',
                                t_eval=time_rom_j,
                                args=(fz, fcdot, const_j, PhiF_dx, Psi) + tensors).y

                sol_xi_j = np.fliplr(sol_xi_j_flipped)
                xi0_j = sol_xi_j[:,0]
                sol_z_j = np.fliplr(sol_z_j_flipped)
                
                # Interpolate z_j and xi onto Gauss-Legendre points
                a = (tf_j - t0_j)/2
                b = (tf_j + t0_j)/2
                time_j_lg = a*tlg + b
                
                fz = sp.interpolate.interp1d(time_rom_j,sol_z_j,kind='linear',fill_value='extrapolate')
                fxi = sp.interpolate.interp1d(time_rom_j,sol_xi_j,kind='linear',fill_value='extrapolate')
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
                    Int_eta_cdot_u0dxx_outer_z_j_cdot_denom += a*wlg[i]*eta_lg[i]*cdot_lg[i] * np.outer(u0_dxx_weighted, z_j_lg[:,i]) / cdot_denom_lg[i]
                    grad_Phi += Psi_dx @ Int_cdot_xi_z_j - Int_eta_cdot_u0dxx_outer_z_j_cdot_denom

                    grad_Psi -= PhiF_dx @ Int_cdot_xi_z_j.T
                    grad_Psi += PhiF @ Int_cdot_xi_z_j.T @ (Psi.T @ PhiF_dx)
                    grad_Psi += PhiF @ (Int_eta_cdot_u0dxx_outer_z_j_cdot_denom.T @ PhiF)

                    for (count,p) in enumerate(opt_obj.poly_comp):
                        equation = ','.join(ascii[:p+1])
                        operands = [xi_lg[:,i]] + [z_j_lg[:,i] for _ in range (p)]
                        grad_tensors_trainable[count] -= a*wlg[i]*np.einsum(equation,*operands)
                        equation = ','.join(ascii[:p])
                        operands = [z_j_lg[:,i] for _ in range (p)]
                        grad_tensors_trainable[count + len(opt_obj.poly_comp)] += a*wlg[i]*np.einsum(equation,*operands) * eta_lg[i] / cdot_denom_lg[i]
            
            # Add the term j = 0 in the sums (2.13) and (2.14). Also add the  
            # contribution of the initial condition (last term in (2.14)) to grad_Psi.
            # Add also the contribution of the steady forcing to grad_Psi.            
            e_X_fitted_weighted_0 = e_X_fitted_weighted[:,0]
            z_0            = Z[:,0]
            e_cdot_0       = e_cdot[0]
            cdot_0         = cdot[0]
            
            cdot_denom_0     = opt_obj.compute_shift_speed_denom(z_0, tensors)
            u0dxx_z_0_outer_0 = np.outer(u0_dxx_weighted, z_0)
            
            weight_X_0  = opt_obj.weights_X[k, 0]
            weight_cdot_0 = opt_obj.weights_cdot[k, 0]
            
            xi0_j += 2*weight_X_0*PhiF.T@e_X_fitted_weighted_0
            xi0_j += 2*weight_cdot_0*e_cdot_0*opt_obj.evaluate_shift_speed_adjoint(z_0, cdot_0, *tensors)

            grad_Phi -= 2 * weight_X_0 * np.einsum('i,j', e_X_fitted_weighted_0, z_0)
            grad_Phi -= 2 * weight_cdot_0 * e_cdot_0 * cdot_0 * u0dxx_z_0_outer_0 / cdot_denom_0
            grad_Psi += 2 * weight_X_0 * np.einsum('i,j',PhiF@z_0, PhiF.T@e_X_fitted_weighted_0)
            grad_Psi += 2 * weight_cdot_0 * e_cdot_0 * cdot_0 * PhiF @ (u0dxx_z_0_outer_0.T @ PhiF) / cdot_denom_0
            grad_Psi -= np.einsum('i,j', opt_obj.X_fitted_weighted[k,:,0], xi0_j)
            
            # Project gradients onto the tangent space            
            grad_Phi = (grad_Phi - Psi @ (PhiF.T @ grad_Phi)) @ F.T
        
        # Compute the gradient of the stability-promoting term
        if opt_obj.l2_pen != None and mpi_pool.rank == 0:
            
            idx = opt_obj.poly_comp.index(1)    # index of the linear tensor
            
            A = tensors[idx]
            
            time_pen = np.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom)
            Z = (solve_ivp(lambda t,z: A@z if np.linalg.norm(z) < 1e4 else 0*z,\
                           [0,time_pen[-1]],opt_obj.randic,method='RK45',t_eval=time_pen)).y
            Mu = (solve_ivp(lambda t,z: A.T@z if np.linalg.norm(z) < 1e4 else 0*z,\
                           [0,time_pen[-1]],-2*opt_obj.l2_pen*Z[:,-1],method='RK45',t_eval=time_pen)).y
            Mu = np.fliplr(Mu)
            
            
            for k in range (opt_obj.n_snapshots - 1):
                
                k0, k1 = k*opt_obj.nsave_rom, (k+1)*opt_obj.nsave_rom
                fZ = sp.interpolate.interp1d(time_pen[k0:k1],Z[:,k0:k1],kind='linear',fill_value='extrapolate')
                fMu = sp.interpolate.interp1d(time_pen[k0:k1],Mu[:,k0:k1],kind='linear',fill_value='extrapolate')
            
                a = (time_pen[k1-1] - time_pen[k0])/2
                b = (time_pen[k1-1] + time_pen[k0])/2
                time_k_lg = a*tlg + b
                
                Zk = fZ(time_k_lg)
                Muk = fMu(time_k_lg)
                
                for i in range (opt_obj.leggauss_deg):
                    grad_tensors_trainable[idx] += -a*wlg[i]*np.einsum('i,j',Muk[:,i],Zk[:,i])
                
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




