import numpy as np
import scipy
from scipy.fftpack import shift

def freq_to_space(uhat):

    """Transform frequency space representation to physical space."""
    
    N = len(uhat) // 2
    return np.fft.ifft(np.fft.ifftshift(uhat)).real * 2 * N

def space_to_freq(u):

    """Transform physical space representation to frequency space."""
    
    N = len(u) // 2
    uhat = np.fft.fftshift(np.fft.fft(u)) / (2 * N)
    # if there are even number of Fourier modes, we will only have 1 Nyquist frequency mode
    # to keep the solution real-valued, we truncate this Nyquist mode
    if len(uhat) % 2 == 0:
        uhat[0] = 0
    return uhat

def compute_shift_speed_FOM(rhs_fitted, sol_fitted_dx, sol_template_dx):
    """Compute the shifting speed for the FOM."""
    return 

class KSE:

    def __init__(self, L, nu, nx, u_template = None, u_template_dx = None):
        self.L = L
        self.nu = nu
        self.nx = nx
        self.nmodes = nx // 2
        self.mode_idx = np.linspace(-self.nmodes, self.nmodes-1, 2*self.nmodes, dtype=int, endpoint=True)
        self.k = 2 * np.pi * self.mode_idx / L
        self._deriv_factor = 1j * self.k
        self._linear_factor = -self.nu * self._deriv_factor ** 4 - self._deriv_factor ** 2

        self.u_template = u_template
        self.u_template_dx = u_template_dx

    def inner_product(self, u, v):
        """Compute the inner product of two spatial functions u and v."""
        dx = self.L / self.nx
        return np.dot(u, v) * dx
    
    def outer_product(self, u, v):
        """Compute the outer product of two spatial functions u and v."""
        dx = self.L / self.nx
        return np.einsum('i,j', u, v) * dx

    def shift(self, u, c):

        """Apply shift operation on u: 
        Given a spatial function u(x), returns S_c[u](x) = u(x - c)
        
        args:
            u(x): spatial function (1D array)
            c: shift amount (float)
            L: domain length (float)
        returns:
            u(x-c): shifted spatial function (1D array)
        """
        
        uhat = space_to_freq(u)
        N = self.nx // 2
        idx  = np.linspace(-N, N-1, 2*N, dtype=int, endpoint=True)
        uhat_shifted = uhat * np.exp(2j * np.pi * idx * (-c) / self.L)
        
        return freq_to_space(uhat_shifted)

    def template_fitting(self, sol, c_old = None):
        
        if sol.ndim == 1 or sol.shape[1] == 1: # this means that we are dealing with a single snapshot
            if c_old is None: # this means we are fitting the initial condition, we need a larger search range
                c_range = np.linspace(-self.L/2, self.L/2, 10000 * self.nx, endpoint=False)
                minimal_error = 1e5
                for c in c_range:
                    sol_fitted_tmp = self.shift(sol, -c) # minimize the difference between u(x + c) and the template
                    error = np.linalg.norm(sol_fitted_tmp - self.u_template) # take the initial condition as the template
                    # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs
                    if error < minimal_error:
                        minimal_error = error
                        c_new = c
                return self.shift(sol, -c_new), c_new
            else: # this means we are fitting the following snapshots, we only need to modify the old shifting amount by a small amount
                dx = self.L / self.nx
                c_step_range = np.linspace(-1 * dx, 1 * dx, 10 * self.nx, endpoint=False)
                minimal_error = 1e5
                for c_step in c_step_range:
                    sol_fitted_tmp = self.shift(sol, -(c_old + c_step))
                    error = np.linalg.norm(sol_fitted_tmp - self.u_template) # take the initial condition as the template
                    # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs
                    if error < minimal_error:
                        minimal_error = error
                        c_new = c_old + c_step
                return self.shift(sol, -c_new), c_new

        else: # this means we are dealing with a trajectory matrix of possible the form (nx, nt)
            n_snapshots = sol.shape[1]
            dx = self.L / self.nx
            c = np.zeros(n_snapshots)
            c_range = np.linspace(-self.L/2, self.L/2, 10000 * self.nx, endpoint=False)
            c_step_range = np.linspace(-1 * dx, 1 * dx, 10 * self.nx, endpoint=False)
            sol_fitted = np.zeros_like(sol)
            minimal_error = 1e5

            for c_init in c_range:

                sol_slice_fitted_tmp = self.shift(sol[:, 0], -c_init)
                error = np.linalg.norm(sol_slice_fitted_tmp - self.u_template)
                if error < minimal_error:
                    minimal_error = error
                    c[0] = c_init

            sol_fitted[:, 0] = self.shift(sol[:, 0], -c[0])

            for time in range(1, n_snapshots):

                minimal_error = 1e5

                c[time] = c[time - 1]

                for c_step in c_step_range:

                    sol_slice_fitted_tmp = self.shift(sol[:, time], -(c[time - 1] + c_step))

                    error = np.linalg.norm(sol_slice_fitted_tmp - self.u_template) # take the initial condition as the template
                    # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs

                    if error < minimal_error:
                        minimal_error = error
                        c[time] = c[time - 1] + c_step

                sol_fitted[:, time] = self.shift(sol[:, time], -c[time])

            return sol_fitted, c

    def take_derivative(self, u, order):

        """Take spatial derivative of order 'order' of u(x).
        
        args:
            u: spatial function (1D or 2D array)
            order: order of derivative (int)
        returns:
            d^order u / dx^order: spatial derivative (1D or 2D array)
            if u is 2D, derivative is taken column-wise (Nx by Nt array)
        """
        if u.ndim == 1:
            return freq_to_space(self._deriv_factor ** order * space_to_freq(u))
        elif u.ndim == 2:
            return np.apply_along_axis(lambda ucol: freq_to_space(self._deriv_factor ** order * space_to_freq(ucol)), axis=0, arr=u)
    
    def linear(self, u):

        """Evaluate linear operator L(u) = -nu * u_xxxx - u_xx."""
        return freq_to_space(self._linear_factor * space_to_freq(u))

    def bilinear(self, u, v):

        """Evaluate bilinear operator B(u, v) = -(udv + vdu)/2. via spectral Galerkin discretization."""
        bilinear_udv = np.zeros_like(u, dtype=complex)
        bilinear_vdu = np.zeros_like(u, dtype=complex)
        
        uhat = space_to_freq(u)
        vhat = space_to_freq(v)
        
        for p in range(self.nx):
            freq_p = self.mode_idx[p]
            for m in range(self.nx):
                freq_m = self.mode_idx[m]
                freq_n = freq_p - freq_m
                n = freq_n + self.nmodes
                if freq_n <= self.nmodes - 1 and freq_n >= -self.nmodes:
                    bilinear_udv[p] += -self._deriv_factor[m] * uhat[n] * vhat[m]
                    bilinear_vdu[p] += -self._deriv_factor[m] * vhat[n] * uhat[m]
        
        bilinear_udv[0] = 0
        bilinear_vdu[0] = 0
        return freq_to_space((bilinear_udv + bilinear_vdu) / 2)
    
    def nonlinear(self, u):
        """Evaluate nonlinear operator N(u) = B(u, u)."""
        return self.bilinear(u, u)
    
    def evaluate_fom_rhs(self, t, u, forcing):
        f_ext = forcing.copy() if hasattr(forcing,"__len__") == True else forcing(t)
        if np.linalg.norm(u) >= 1e4:    rhs = 0*u
        else:                           rhs = self.linear(u) + self.nonlinear(u) + f_ext

        return rhs

    def evaluate_fom_shift_speed(self, rhs_fitted, u_fitted_dx):
        """Evaluate the shift speed for the FOM."""
        cdot_numer = - self.inner_product(rhs_fitted, self.u_template_dx)
        cdot_denom = self.inner_product(u_fitted_dx, self.u_template_dx)
        if np.abs(cdot_denom) < 1e-10:
            raise ValueError("Denominator in computing FOM shift speed is too small!")
        else:
            return cdot_numer / cdot_denom
        
    def evaluate_fom_shift_speed_numer(self, rhs_fitted):
        """Evaluate the shift speed for the FOM."""
        return - self.inner_product(rhs_fitted, self.u_template_dx)
    
    def evaluate_fom_shift_speed_denom(self, u_fitted_dx):
        """Evaluate the shift speed for the FOM."""
        return self.inner_product(u_fitted_dx, self.u_template_dx)

    def assemble_petrov_galerkin_tensors(self, Phi, Psi):
        """Assemble the Petrov-Galerkin projection matrices."""
        
        r = Phi.shape[1]
        A_mat = np.zeros((r, r))
        B_tensor = np.zeros((r, r, r))
        p_vec = np.zeros(r)
        Q_mat = np.zeros((r, r))
        s_vec = np.zeros(r)
        M_mat = np.zeros((r, r))

        PhiF = Phi@scipy.linalg.inv(Psi.T@Phi)
        PhiF_dx = self.take_derivative(PhiF, order=1)
        
        u0_dx = self.u_template_dx
        # u0_dxx = - self.u_template

        M_mat = Psi.T @ PhiF_dx
        for i in range(r):
            p_vec[i] = self.inner_product(u0_dx, self.linear(PhiF[:, i]))
            s_vec[i] = self.inner_product(u0_dx, PhiF_dx[:, i])
            # s_vec[i] = - self.inner_product(u0_dxx, PhiF[:, i])
            for j in range(r):
                A_mat[i, j] = np.dot(Psi[:, i], self.linear(PhiF[:, j]))
                Q_mat[i, j] = self.inner_product(u0_dx, self.bilinear(PhiF[:, i], PhiF[:, j]))
                for k in range(r):
                    B_tensor[i, j, k] = np.dot(Psi[:, i], self.bilinear(PhiF[:, j], PhiF[:, k]))

        return (A_mat, B_tensor, p_vec, Q_mat, s_vec, M_mat)
    
class time_step_kse:
    
    def __init__(self, fom, time):
        self.fom = fom
        self.time = time
        self.dt = time[1] - time[0]
        self.nsteps = len(time)
        
    def get_solver(self, alpha):
        """This returns a function that can be used to solve linear systems encountered in semi-implicit schemes.
           like: (I - alpha*dt*L) x = b  or  (I - alpha*dt*J) x = b
        """
        def solver(rhs):
            return freq_to_space(space_to_freq(rhs)/(1 - alpha * self.fom._linear_factor))
        return solver
        
    def time_step(self, u, nsave, *argv):

        """Time-step the KSE from initial condition q0 using CNRK3 scheme.
        See Peyret (2002) "Spectral Methods for Incompressible Viscous Flow" for details.
        
        args:
            u: initial condition (1D array of length Nx)
            nsave: save every nsave time steps (int)
            *argv: additional arguments for the forcing function (if any)
        returns:
            Qkj: state snapshots (Nx by Nt/nsave array)
            tsave: saved time instances (1D array)
        """
        tsave = self.time[::nsave]
        u_snapshots = np.zeros((self.fom.nx, len(tsave)))
        u_snapshots[:, 0] = u

        A = [0, -5.0/9, -153.0/128]
        B = [1.0/3, 15.0/16, 8.0/15]
        Bprime = [1.0/6, 5.0/24, 1.0/8]
        C = [0, 1.0/3, 3.0/4]

        solvers = [self.get_solver(b * self.dt) for b in Bprime]

        if len(argv) == 0:
            k_save = 1
            for k in range(1, len(self.time)):
                
                nl_1 = self.dt * self.fom.nonlinear(u)
                rhs_1 = u + B[0] * nl_1 + Bprime[0] * self.dt * self.fom.linear(u)
                u1 = solvers[0](rhs_1)

                nl_2 = A[1] * nl_1 + self.dt * self.fom.nonlinear(u1)
                rhs_2 = u1 + B[1] * nl_2 + Bprime[1] * self.dt * self.fom.linear(u1)
                u2 = solvers[1](rhs_2)
                
                nl_3 = A[2] * nl_2 + self.dt * self.fom.nonlinear(u2)
                rhs_3 = u2 + B[2] * nl_3 + Bprime[2] * self.dt * self.fom.linear(u2)
                u = solvers[2](rhs_3)

                if k % nsave == 0:
                    u_snapshots[:, k_save] = u
                    k_save += 1
        else:
            f_ext = argv[0]
            f_ext_period = argv[1]
            k_save = 1
            for k in range(1, len(self.time)):

                nl_1 = self.dt * (self.fom.nonlinear(u) + f_ext(np.mod(self.time[k-1] + C[0] * self.dt, f_ext_period)))
                rhs_1 = u + B[0] * nl_1 + Bprime[0] * self.dt * self.fom.linear(u)
                solver_1 = self.get_solver(Bprime[0] * self.dt)
                u1 = solver_1(rhs_1)

                nl_2 = A[1] * nl_1 + self.dt * (self.fom.nonlinear(u1) + f_ext(np.mod(self.time[k-1] + C[1] * self.dt, f_ext_period)))
                rhs_2 = u1 + B[1] * nl_2 + Bprime[1] * self.dt * self.fom.linear(u1)
                solver_2 = self.get_solver(Bprime[1] * self.dt)
                u2 = solver_2(rhs_2)

                nl_3 = A[2] * nl_2 + self.dt * (self.fom.nonlinear(u2) + f_ext(np.mod(self.time[k-1] + C[2] * self.dt, f_ext_period)))
                rhs_3 = u2 + B[2] * nl_3 + Bprime[2] * self.dt * self.fom.linear(u2)
                solver_3 = self.get_solver(Bprime[2] * self.dt)
                u = solver_3(rhs_3)

                if k % nsave == 0:
                    u_snapshots[:, k_save] = u
                    k_save += 1

        return u_snapshots, tsave