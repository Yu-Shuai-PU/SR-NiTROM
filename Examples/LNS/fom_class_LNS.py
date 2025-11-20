import numpy as np
import scipy
from scipy.fftpack import shift

def freq_to_space_1D(uhat):
    """Transform frequency space representation to physical space in 1D."""
    N = len(uhat)
    return np.fft.ifft(np.fft.ifftshift(uhat)).real * N

def space_to_freq_1D(u):
    """Transform physical space representation to frequency space in 1D."""
    N = len(u)
    uhat = np.fft.fftshift(np.fft.fft(u)) / N
    # if there are even number of Fourier modes, we will only have 1 Nyquist frequency mode
    # to keep the solution real-valued, we truncate this Nyquist mode
    if N % 2 == 0:
        uhat[0] = 0
    return uhat

def space_to_freq_2D(u):
    """Transform physical space representation to frequency space in 2D."""
    Nx, Nz = u.shape
    uhat = np.fft.fftshift(np.fft.fft2(u)) / (Nx * Nz)
    if Nx % 2 == 0:
        uhat[0, :] = 0
    if Nz % 2 == 0:
        uhat[:, 0] = 0
    return uhat

def freq_to_space_2D(uhat):
    """Transform frequency space representation to physical space in 2D."""
    Nx, Nz = uhat.shape
    return np.fft.ifft2(np.fft.ifftshift(uhat)).real * (Nx * Nz)

def compute_shift_speed_FOM(rhs_fitted, sol_fitted_dx, sol_template_dx):
    """Compute the shifting speed for the FOM."""
    return

def Chebyshev_diff_mat(y):
    
    ### Generate Chebyshev differentiation matrix for N grid points in [-1, 1] including boundaries.
    D = np.zeros((len(y), len(y)))
    N = len(y) - 1
    D[0, 0] = (2 * N ** 2 + 1) / 6
    D[N, N] = -D[0, 0]
    for j in range(1, N):
        D[j, j] = - y[j] / (2 * (1 - y[j] ** 2))
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                ci = 2 if (i == 0 or i == N) else 1
                cj = 2 if (j == 0 or j == N) else 1
                D[i, j] = (ci / cj) * ((-1) ** (i + j)) / (y[i] - y[j])
    
    return D 

def Chebyshev_diff_FFT(f):
    y = np.cos(np.pi * np.linspace(0, len(f) - 1, num=len(f)) / (len(f) - 1))  # Chebyshev grid in y direction, location from 1 to -1
    N = len(f) - 1
    ### Compute the Chebyshev derivative of f using FFT method.
    ### Input data: f from x_0 = 1 to x_N = -1 (N+1 points)
    ### See Trefethen 2000, Spectral Methods in MATLAB, p.80
    # Step 1: Extend f to a 2N periodic function
    F = np.zeros(2 * N)
    F[0:N+1] = f[0:N+1]
    F[N+1:2*N] = f[N-1:0:-1]
    
    # Step 2: Compute the FFT of F with wavenumbers
    Fhat = np.fft.fft(F).real
    k = np.fft.fftfreq(2 * N, d = 1 / (2 * N))
    # Step 3: Truncate the Nyquist frequency mode
    k[N] = 0
    # Step 4: Compute the derivative in frequency space
    Fhat_dx = 1j * k * Fhat
    # Step 5: Transform back to physical space
    F_dx = np.fft.ifft(Fhat_dx).real
    # Step 6: Extract the derivative values at the original Chebyshev points
    f_dx = np.zeros(N + 1)
    for i in range(1, N):
        f_dx[i] = -F_dx[i] / np.sqrt(1 - (y[i]) ** 2)
        f_dx[0] += i**2 * Fhat[i] / N
        f_dx[N] += (-1)**(i+1) * i**2 * Fhat[i] / N
    f_dx[0] += 0.5 * N * Fhat[N]
    f_dx[N] += 0.5 * (-1) ** (N + 1) * N * Fhat[N]
    return f_dx

class LNS:
    
    def __init__(self, Lx, Ly, Lz, nx, ny, nz, y, Re, U_base, U_base_dy, U_base_dyy):
        
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Re = Re
        
        self.y = y # Chebyshev grid in y direction, location from 1 to -1
        self.U_base = U_base  # base flow U(y)
        self.U_base_dy = U_base_dy  # derivative of base flow U(y)
        self.U_base_dyy = U_base_dyy  # second derivative of base flow U(y)
        
        self.U_base_mat = np.diag(U_base[1:-1])
        self.U_base_dy_mat = np.diag(U_base_dy[1:-1])
        self.U_base_dyy_mat = np.diag(U_base_dyy[1:-1])

        self.nmodes_x = nx // 2
        self.nmodes_z = nz // 2
        self.mode_idx_x = np.linspace(-self.nmodes_x, self.nmodes_x-1, 2*self.nmodes_x, dtype=int, endpoint=True)
        self.mode_idx_z = np.linspace(-self.nmodes_z, self.nmodes_z-1, 2*self.nmodes_z, dtype=int, endpoint=True)
        self.kx = 2 * np.pi * self.mode_idx_x / Lx
        self.kz = 2 * np.pi * self.mode_idx_z / Lz
        self._deriv_factor_x = 1j * self.kx
        self._deriv_factor_z = 1j * self.kz
        self._deriv_factor_x = 1j * self.kx
        
    def Fourier_diff_x(self, u, order):
        """Take spatial derivative of order 'order' of u(x).
        
        args:
            u: spatial function (1D or 2D array)
            order: order of derivative (int)
        returns:
            d^order u / dx^order: spatial derivative (1D or 2D array)
            if u is 2D, derivative is taken column-wise (Nx by Nt array)
        """
        if u.ndim == 1:
            return freq_to_space_1D(self._deriv_factor_x ** order * space_to_freq_1D(u))
        elif u.ndim == 2:
            return np.apply_along_axis(lambda ucol: freq_to_space_1D(self._deriv_factor_x ** order * space_to_freq_1D(ucol)), axis=0, arr=u)

    def Fourier_diff_z(self, u, order):
        """Take spatial derivative of order 'order' of u(z).
        
        args:
            u: spatial function (1D or 2D array)
            order: order of derivative (int)
        returns:
            d^order u / dx^order: spatial derivative (1D or 2D array)
            if u is 2D, derivative is taken column-wise (Nx by Nt array)
        """
        if u.ndim == 1:
            return freq_to_space_1D(self._deriv_factor_z ** order * space_to_freq_1D(u))
        elif u.ndim == 2:
            return np.apply_along_axis(lambda ucol: freq_to_space_1D(self._deriv_factor_z ** order * space_to_freq_1D(ucol)), axis=0, arr=u)

    def Fourier_transform_xz(self, u):
        """Perform 2D Fourier transform in x and z directions.
           Input u(x, y, z) with shape (nx, ny, nz)
           Output uhat(kx, y, kz) with shape (nx, ny, nz)
        """
        uhat = np.zeros((self.nx, self.ny, self.nz), dtype=complex)
        for idx_y in range(self.ny):
            u_slice = u[:, idx_y, :]
            uhat[:, idx_y, :] = space_to_freq_2D(u_slice)
            
        return uhat
    
    def Fourier_inverse_transform_xz(self, uhat):
        """Perform inverse 2D Fourier transform in x and z directions.
           Input uhat(kx, y, kz) with shape (nx, ny, nz)
           Output u(x, y, z) with shape (nx, ny, nz)
        """
        u = np.zeros((self.nx, self.ny, self.nz))
        for idx_y in range(self.ny):
            uhat_slice = uhat[:, idx_y, :]
            u[:, idx_y, :] = freq_to_space_2D(uhat_slice)
            
        return u
    
    def apply_BC(self, D1):
        """Apply boundary conditions to the Chebyshev differentiation matrix D1.
           v = dv/dy = eta = 0 at y = 1 and -1
           see Trefethen 2000, Spectral Methods in MATLAB, p. 146 for details
        """
        y = np.cos(np.pi * np.linspace(0, D1.shape[0] - 1, num=D1.shape[0]) / (D1.shape[0] - 1))  # Chebyshev grid in y direction, location from 1 to -1
        N = D1.shape[0] - 1
        D2 = D1 @ D1
        D2 = D2[1:N, 1:N]
        coeff_vec = np.concatenate(([0], 1/(1 - y[1:N] ** 2), [0]))
        coeff_mat = np.diag(coeff_vec)
        D4 = (np.diag(1 - y**2) @ D1 @ D1 @ D1 @ D1
             -8 * np.diag(y) @ D1 @ D1 @ D1
             -12 * D1 @ D1) @ coeff_mat
        D4 = D4[1:N, 1:N]
        return D2, D4
    
    """
    Below is the formal form of the LNS class, where we seek to represent the dynamics as
    
    dq/dt = Lq + f_ext.
    where 
    q(idx_x * (Ny * Nz) + idx_y * Nz + idx_z)                = v(idx_x, idx_y, idx_z) for 0 <= idx_x <= Nx - 1, 0 <= idx_y <= Ny - 1, 0 <= idx_z <= Nz - 1
    q(Nx * Ny * Nz + idx_x * (Ny * Nz) + idx_y * Nz + idx_z) = eta(idx_x, idx_y, idx_z) for 0 <= idx_x <= Nx - 1, 0 <= idx_y <= Ny - 1, 0 <= idx_z <= Nz - 1      
    and L is the linear operator representing the linearized Navier-Stokes equations around the base flow
    """
    
    def linear(self, q_1D):
        """Evaluate linear operator Lq.
        We apply the linear operator in 3 steps
        1. Transform u(x,y,z) to uhat(kx,y,kz) via 2D Fourier transform in x and z directions
        2. For each (kx, kz) mode, apply the linear operator in y direction via differentiation matrices obtained from Chebyshev collocation method w/ BCs
        3. Transform back to physical space u(x,y,z) via inverse 2D Fourier transform in x and z directions
        """
        v = q_1D[0 : self.nx * self.ny * self.nz].reshape((self.nx, self.ny, self.nz))
        eta = q_1D[self.nx * self.ny * self.nz : ].reshape((self.nx, self.ny, self.nz))
        
        v_hat = self.Fourier_transform_xz(v)
        eta_hat = self.Fourier_transform_xz(eta)
        
        D1 = Chebyshev_diff_mat(self.y)
        D2, D4 = self.apply_BC(D1)
        Id = np.eye(self.ny - 2, dtype=complex)
        M = np.zeros((2 * (self.ny - 2), 2 * (self.ny - 2)), dtype=complex)
        L = np.zeros((2 * (self.ny - 2), 2 * (self.ny - 2)), dtype=complex)
        state = np.zeros((2 * (self.ny - 2)), dtype=complex)

        linear_v_hat = np.zeros((self.nx, self.ny, self.nz), dtype=complex)
        linear_eta_hat = np.zeros((self.nx, self.ny, self.nz), dtype=complex)

        for idx_kx in range(self.nx):
            kx = self.kx[idx_kx]
            for idx_kz in range(self.nz):
                kz = self.kz[idx_kz]
                Laplace = D2 - (kx**2 + kz**2) * Id
                Bi_Laplace = D4 - 2 * (kx**2 + kz**2) * D2 + (kx**2 + kz**2)**2 * Id 
                M[:self.ny - 2, :self.ny - 2] = Laplace
                M[self.ny - 2:, self.ny - 2:] = Id
                L[:self.ny - 2, :self.ny - 2] = -1j * kx * (self.U_base_mat @ Laplace) + 1j * kx * self.U_base_dyy_mat + Bi_Laplace / self.Re
                L[self.ny - 2:, :self.ny - 2] = -1j * kz * self.U_base_dy_mat
                L[self.ny - 2:, self.ny - 2:] = -1j * kx * self.U_base_mat + Laplace / self.Re
                linear_mat = scipy.linalg.solve(M, L)
                state[:self.ny-2] = v_hat[idx_kx, 1:-1, idx_kz]
                state[self.ny-2:] = eta_hat[idx_kx, 1:-1, idx_kz]
                linear_state_hat = linear_mat @ state
                linear_v_hat[idx_kx, 1:-1, idx_kz] = linear_state_hat[:self.ny-2]
                linear_eta_hat[idx_kx, 1:-1, idx_kz] = linear_state_hat[self.ny-2:]
                
        linear_v = self.Fourier_inverse_transform_xz(linear_v_hat)
        linear_eta = self.Fourier_inverse_transform_xz(linear_eta_hat)
        linear_q_1D = np.concatenate((linear_v.ravel(), linear_eta.ravel()))
        
        return linear_q_1D

# class KSE:

#     def __init__(self, L, nu, nx, u_template = None, u_template_dx = None):
#         self.L = L
#         self.nu = nu
#         self.nx = nx
#         self.nmodes = nx // 2
#         self.mode_idx = np.linspace(-self.nmodes, self.nmodes-1, 2*self.nmodes, dtype=int, endpoint=True)
#         self.k = 2 * np.pi * self.mode_idx / L
#         self._deriv_factor = 1j * self.k
#         self._linear_factor = -self.nu * self._deriv_factor ** 4 - self._deriv_factor ** 2

#         self.u_template = u_template
#         self.u_template_dx = u_template_dx

#     def inner_product(self, u, v):
#         """Compute the inner product of two spatial functions u and v."""
#         dx = self.L / self.nx
#         return np.dot(u, v) * dx
    
#     def outer_product(self, u, v):
#         """Compute the outer product of two spatial functions u and v."""
#         dx = self.L / self.nx
#         return np.einsum('i,j', u, v) * dx

#     def shift(self, u, c):

#         """Apply shift operation on u: 
#         Given a spatial function u(x), returns S_c[u](x) = u(x - c)
        
#         args:
#             u(x): spatial function (1D array)
#             c: shift amount (float)
#             L: domain length (float)
#         returns:
#             u(x-c): shifted spatial function (1D array)
#         """
        
#         uhat = space_to_freq(u)
#         N = self.nx // 2
#         idx  = np.linspace(-N, N-1, 2*N, dtype=int, endpoint=True)
#         uhat_shifted = uhat * np.exp(2j * np.pi * idx * (-c) / self.L)
        
#         return freq_to_space(uhat_shifted)

#     def template_fitting(self, sol, c_old = None):
        
#         if sol.ndim == 1 or sol.shape[1] == 1: # this means that we are dealing with a single snapshot
#             if c_old is None: # this means we are fitting the initial condition, we need a larger search range
#                 c_range = np.linspace(-self.L/2, self.L/2, 10000 * self.nx, endpoint=False)
#                 minimal_error = 1e5
#                 for c in c_range:
#                     sol_fitted_tmp = self.shift(sol, -c) # minimize the difference between u(x + c) and the template
#                     error = np.linalg.norm(sol_fitted_tmp - self.u_template) # take the initial condition as the template
#                     # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs
#                     if error < minimal_error:
#                         minimal_error = error
#                         c_new = c
#                 return self.shift(sol, -c_new), c_new
#             else: # this means we are fitting the following snapshots, we only need to modify the old shifting amount by a small amount
#                 dx = self.L / self.nx
#                 c_step_range = np.linspace(-1 * dx, 1 * dx, 10 * self.nx, endpoint=False)
#                 minimal_error = 1e5
#                 for c_step in c_step_range:
#                     sol_fitted_tmp = self.shift(sol, -(c_old + c_step))
#                     error = np.linalg.norm(sol_fitted_tmp - self.u_template) # take the initial condition as the template
#                     # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs
#                     if error < minimal_error:
#                         minimal_error = error
#                         c_new = c_old + c_step
#                 return self.shift(sol, -c_new), c_new

#         else: # this means we are dealing with a trajectory matrix of possible the form (nx, nt)
#             n_snapshots = sol.shape[1]
#             dx = self.L / self.nx
#             c = np.zeros(n_snapshots)
#             c_range = np.linspace(-self.L/2, self.L/2, 10000 * self.nx, endpoint=False)
#             c_step_range = np.linspace(-1 * dx, 1 * dx, 10 * self.nx, endpoint=False)
#             sol_fitted = np.zeros_like(sol)
#             minimal_error = 1e5

#             for c_init in c_range:

#                 sol_slice_fitted_tmp = self.shift(sol[:, 0], -c_init)
#                 error = np.linalg.norm(sol_slice_fitted_tmp - self.u_template)
#                 if error < minimal_error:
#                     minimal_error = error
#                     c[0] = c_init

#             sol_fitted[:, 0] = self.shift(sol[:, 0], -c[0])

#             for time in range(1, n_snapshots):

#                 minimal_error = 1e5

#                 c[time] = c[time - 1]

#                 for c_step in c_step_range:

#                     sol_slice_fitted_tmp = self.shift(sol[:, time], -(c[time - 1] + c_step))

#                     error = np.linalg.norm(sol_slice_fitted_tmp - self.u_template) # take the initial condition as the template
#                     # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs

#                     if error < minimal_error:
#                         minimal_error = error
#                         c[time] = c[time - 1] + c_step

#                 sol_fitted[:, time] = self.shift(sol[:, time], -c[time])

#             return sol_fitted, c

#     def take_derivative(self, u, order):

#         """Take spatial derivative of order 'order' of u(x).
        
#         args:
#             u: spatial function (1D or 2D array)
#             order: order of derivative (int)
#         returns:
#             d^order u / dx^order: spatial derivative (1D or 2D array)
#             if u is 2D, derivative is taken column-wise (Nx by Nt array)
#         """
#         if u.ndim == 1:
#             return freq_to_space(self._deriv_factor ** order * space_to_freq(u))
#         elif u.ndim == 2:
#             return np.apply_along_axis(lambda ucol: freq_to_space(self._deriv_factor ** order * space_to_freq(ucol)), axis=0, arr=u)
    
#     def linear(self, u):

#         """Evaluate linear operator L(u) = -nu * u_xxxx - u_xx."""
#         return freq_to_space(self._linear_factor * space_to_freq(u))

#     def bilinear(self, u, v):

#         """Evaluate bilinear operator B(u, v) = -(udv + vdu)/2. via spectral Galerkin discretization."""
#         bilinear_udv = np.zeros_like(u, dtype=complex)
#         bilinear_vdu = np.zeros_like(u, dtype=complex)
        
#         uhat = space_to_freq(u)
#         vhat = space_to_freq(v)
        
#         for p in range(self.nx):
#             freq_p = self.mode_idx[p]
#             for m in range(self.nx):
#                 freq_m = self.mode_idx[m]
#                 freq_n = freq_p - freq_m
#                 n = freq_n + self.nmodes
#                 if freq_n <= self.nmodes - 1 and freq_n >= -self.nmodes:
#                     bilinear_udv[p] += -self._deriv_factor[m] * uhat[n] * vhat[m]
#                     bilinear_vdu[p] += -self._deriv_factor[m] * vhat[n] * uhat[m]
        
#         bilinear_udv[0] = 0
#         bilinear_vdu[0] = 0
#         return freq_to_space((bilinear_udv + bilinear_vdu) / 2)
    
#     def nonlinear(self, u):
#         """Evaluate nonlinear operator N(u) = B(u, u)."""
#         return self.bilinear(u, u)
    
#     def evaluate_fom_rhs(self, t, u, forcing):
#         f_ext = forcing.copy() if hasattr(forcing,"__len__") == True else forcing(t)
#         if np.linalg.norm(u) >= 1e4:    rhs = 0*u
#         else:                           rhs = self.linear(u) + self.nonlinear(u) + f_ext

#         return rhs

#     def evaluate_fom_shift_speed(self, rhs_fitted, u_fitted_dx):
#         """Evaluate the shift speed for the FOM."""
#         cdot_numer = - self.inner_product(rhs_fitted, self.u_template_dx)
#         cdot_denom = self.inner_product(u_fitted_dx, self.u_template_dx)
#         if np.abs(cdot_denom) < 1e-10:
#             raise ValueError("Denominator in computing FOM shift speed is too small!")
#         else:
#             return cdot_numer / cdot_denom
        
#     def evaluate_fom_shift_speed_numer(self, rhs_fitted):
#         """Evaluate the shift speed for the FOM."""
#         return - self.inner_product(rhs_fitted, self.u_template_dx)
    
#     def evaluate_fom_shift_speed_denom(self, u_fitted_dx):
#         """Evaluate the shift speed for the FOM."""
#         return self.inner_product(u_fitted_dx, self.u_template_dx)

#     def assemble_petrov_galerkin_tensors(self, Phi, Psi):
#         """Assemble the Petrov-Galerkin projection matrices."""
        
#         r = Phi.shape[1]
#         A_mat = np.zeros((r, r))
#         B_tensor = np.zeros((r, r, r))
#         p_vec = np.zeros(r)
#         Q_mat = np.zeros((r, r))
#         s_vec = np.zeros(r)
#         M_mat = np.zeros((r, r))

#         PhiF = Phi@scipy.linalg.inv(Psi.T@Phi)
#         PhiF_dx = self.take_derivative(PhiF, order=1)
        
#         u0_dx = self.u_template_dx
#         # u0_dxx = - self.u_template

#         M_mat = Psi.T @ PhiF_dx
#         for i in range(r):
#             p_vec[i] = self.inner_product(u0_dx, self.linear(PhiF[:, i]))
#             s_vec[i] = self.inner_product(u0_dx, PhiF_dx[:, i])
#             # s_vec[i] = - self.inner_product(u0_dxx, PhiF[:, i])
#             for j in range(r):
#                 A_mat[i, j] = np.dot(Psi[:, i], self.linear(PhiF[:, j]))
#                 Q_mat[i, j] = self.inner_product(u0_dx, self.bilinear(PhiF[:, i], PhiF[:, j]))
#                 for k in range(r):
#                     B_tensor[i, j, k] = np.dot(Psi[:, i], self.bilinear(PhiF[:, j], PhiF[:, k]))

#         return (A_mat, B_tensor, p_vec, Q_mat, s_vec, M_mat)
    
# class time_step_kse:
    
    # def __init__(self, fom, time):
    #     self.fom = fom
    #     self.time = time
    #     self.dt = time[1] - time[0]
    #     self.nsteps = len(time)
        
    # def get_solver(self, alpha):
    #     """This returns a function that can be used to solve linear systems encountered in semi-implicit schemes.
    #        like: (I - alpha*dt*L) x = b  or  (I - alpha*dt*J) x = b
    #     """
    #     def solver(rhs):
    #         return freq_to_space(space_to_freq(rhs)/(1 - alpha * self.fom._linear_factor))
    #     return solver
        
    # def time_step(self, u, nsave, *argv):

    #     """Time-step the KSE from initial condition q0 using CNRK3 scheme.
    #     See Peyret (2002) "Spectral Methods for Incompressible Viscous Flow" for details.
        
    #     args:
    #         u: initial condition (1D array of length Nx)
    #         nsave: save every nsave time steps (int)
    #         *argv: additional arguments for the forcing function (if any)
    #     returns:
    #         Qkj: state snapshots (Nx by Nt/nsave array)
    #         tsave: saved time instances (1D array)
    #     """
    #     tsave = self.time[::nsave]
    #     u_snapshots = np.zeros((self.fom.nx, len(tsave)))
    #     u_snapshots[:, 0] = u

    #     A = [0, -5.0/9, -153.0/128]
    #     B = [1.0/3, 15.0/16, 8.0/15]
    #     Bprime = [1.0/6, 5.0/24, 1.0/8]
    #     C = [0, 1.0/3, 3.0/4]

    #     solvers = [self.get_solver(b * self.dt) for b in Bprime]

    #     if len(argv) == 0:
    #         k_save = 1
    #         for k in range(1, len(self.time)):
                
    #             nl_1 = self.dt * self.fom.nonlinear(u)
    #             rhs_1 = u + B[0] * nl_1 + Bprime[0] * self.dt * self.fom.linear(u)
    #             u1 = solvers[0](rhs_1)

    #             nl_2 = A[1] * nl_1 + self.dt * self.fom.nonlinear(u1)
    #             rhs_2 = u1 + B[1] * nl_2 + Bprime[1] * self.dt * self.fom.linear(u1)
    #             u2 = solvers[1](rhs_2)
                
    #             nl_3 = A[2] * nl_2 + self.dt * self.fom.nonlinear(u2)
    #             rhs_3 = u2 + B[2] * nl_3 + Bprime[2] * self.dt * self.fom.linear(u2)
    #             u = solvers[2](rhs_3)

    #             if k % nsave == 0:
    #                 u_snapshots[:, k_save] = u
    #                 k_save += 1
    #     else:
    #         f_ext = argv[0]
    #         f_ext_period = argv[1]
    #         k_save = 1
    #         for k in range(1, len(self.time)):

    #             nl_1 = self.dt * (self.fom.nonlinear(u) + f_ext(np.mod(self.time[k-1] + C[0] * self.dt, f_ext_period)))
    #             rhs_1 = u + B[0] * nl_1 + Bprime[0] * self.dt * self.fom.linear(u)
    #             solver_1 = self.get_solver(Bprime[0] * self.dt)
    #             u1 = solver_1(rhs_1)

    #             nl_2 = A[1] * nl_1 + self.dt * (self.fom.nonlinear(u1) + f_ext(np.mod(self.time[k-1] + C[1] * self.dt, f_ext_period)))
    #             rhs_2 = u1 + B[1] * nl_2 + Bprime[1] * self.dt * self.fom.linear(u1)
    #             solver_2 = self.get_solver(Bprime[1] * self.dt)
    #             u2 = solver_2(rhs_2)

    #             nl_3 = A[2] * nl_2 + self.dt * (self.fom.nonlinear(u2) + f_ext(np.mod(self.time[k-1] + C[2] * self.dt, f_ext_period)))
    #             rhs_3 = u2 + B[2] * nl_3 + Bprime[2] * self.dt * self.fom.linear(u2)
    #             solver_3 = self.get_solver(Bprime[2] * self.dt)
    #             u = solver_3(rhs_3)

    #             if k % nsave == 0:
    #                 u_snapshots[:, k_save] = u
    #                 k_save += 1

    #     return u_snapshots, tsave