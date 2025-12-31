import numpy as np
import scipy
from scipy.fftpack import shift
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def Clenshaw_Curtis_weights(y):
    """
    From Trefethen 2000, Spectral Methods in MATLAB, p. 128
    """
    # 这里的 N 对应 MATLAB 代码中的输入参数 N
    # 如果 n_points = 5, 那么 N = 4
    N = len(y) - 1 
    if N == 0:
        return np.array([2.0])
    theta = np.pi * np.arange(0, N + 1) / N
    w = np.zeros(N + 1)
    v = np.ones(N - 1)
    theta_ii = theta[1:-1]
    if N % 2 == 0:
        w[0] = 1.0 / (N**2 - 1.0)
        w[-1] = w[0]
        for k in range(1, N // 2): 
            v -= 2.0 * np.cos(2.0 * k * theta_ii) / (4.0 * k**2 - 1.0)
        v -= np.cos(N * theta_ii) / (N**2 - 1.0)
    else:
        w[0] = 1.0 / (N**2)
        w[-1] = w[0]
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2.0 * np.cos(2.0 * k * theta_ii) / (4.0 * k**2 - 1.0)
    w[1:-1] = 2.0 * v / N
    return w

class LNS:
    
    def __init__(self, Lx, Ly, Lz, nx, ny, nz, y, Re, U_base, U_base_dy, U_base_dyy):
        
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Re = Re

        self.x = np.linspace(0, Lx, nx, endpoint=False)
        self.z = np.linspace(0, Lz, nz, endpoint=False)
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
        
        self.k_sq = (self.kx**2)[:, None, None] + (self.kz**2)[None, None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            self.inner_product_factor = 1.0 / (2 * self.k_sq)
        self.inner_product_factor[~np.isfinite(self.inner_product_factor)] = 0.0
        self.inner_product_factor = self.inner_product_factor.squeeze(axis=1)
        
        self.D1 = Chebyshev_diff_mat(y)
        self.D2, self.D4 = self.apply_BC(self.D1)
        
        self.Clenshaw_Curtis_weights = Clenshaw_Curtis_weights(y)
        self.linear_mat = self.assemble_fom_linear_operator()
        self.Cholesky_inner_product_weight()
        
    def apply_BC(self, D1):
        """Apply boundary conditions to the Chebyshev differentiation matrix D1.
           v = dv/dy = eta = 0 at y = 1 and -1
           see Trefethen 2000, Spectral Methods in MATLAB, p. 146 for details
        """  # Chebyshev grid in y direction, location from 1 to -1
        N = D1.shape[0] - 1
        D2 = D1 @ D1
        D2 = D2[1:N, 1:N]
        coeff_vec = np.concatenate(([0], 1/(1 - self.y[1:N] ** 2), [0]))
        coeff_mat = np.diag(coeff_vec)
        D4 = (np.diag(1 - self.y**2) @ D1 @ D1 @ D1 @ D1
             -8 * np.diag(self.y) @ D1 @ D1 @ D1
             -12 * D1 @ D1) @ coeff_mat
        D4 = D4[1:N, 1:N]
        return D2, D4
    
    def FFT_1D(self, u, axis):
        """Compute the 1D FFT along a specified axis of the 3D field u.
        
        Input:
            u -- spatial function with shape (nx, ny, nz)
            axis -- axis along which to compute the FFT (0 for x, 2 for z)
        Output:
            u_tilde -- frequency space representation with shape (nx, ny, nz)
        """
        N = u.shape[axis]
        u_tilde = np.fft.fftshift(np.fft.fft(u, axis=axis), axes=axis) / N
        slices = [slice(None)] * u.ndim # create a dynamic slicer
        slices[axis] = 0 # select the Nyquist frequency mode
        u_tilde[tuple(slices)] *= (N % 2) # if the number of modes is even, set the Nyquist mode to zero
        return u_tilde
    
    def IFFT_1D(self, u_tilde, axis):
        """Compute the 1D inverse FFT along a specified axis of the 3D field u_tilde.
        
        Input:
            u_tilde -- frequency space representation with shape (nx, ny, nz)
            axis -- axis along which to compute the inverse FFT (0 for x, 2 for z)
        Output:
            u -- spatial function with shape (nx, ny, nz)
        """
        N = u_tilde.shape[axis]
        u = np.fft.ifft(np.fft.ifftshift(u_tilde, axes=(axis,)), axis=axis).real * N
        return u
    
    def FFT_2D(self, u):
        """Compute the 2D FFT along x and z axes of the 3D field u.
        (Direct implementation using numpy.fft.fft2)
        
        Input:
            u -- spatial function with shape (nx, ny, nz)
        Output:
            u_breve -- frequency space representation with shape (nx, ny, nz)
        """
        Nx = u.shape[0]
        Nz = u.shape[2]
        u_breve = np.fft.fftshift(np.fft.fft2(u, axes=(0, 2)), axes=(0, 2)) / (Nx * Nz)
        u_breve[0, :, :] *= (Nx % 2)  # if the number of modes is even, set the Nyquist mode to zero
        u_breve[:, :, 0] *= (Nz % 2)
        return u_breve
    
    def IFFT_2D(self, u_breve):
        """Compute the 2D inverse FFT along x and z axes of the 3D field u_breve.
        (Direct implementation using numpy.fft.ifft2)
        
        Input:
            u_breve -- frequency space representation with shape (nx, ny, nz)
        Output:
            u -- spatial function with shape (nx, ny, nz)
        """
        Nx = u_breve.shape[0]
        Nz = u_breve.shape[2]
        return np.fft.ifft2(np.fft.ifftshift(u_breve, axes=(0, 2)), axes=(0, 2)).real * (Nx * Nz)
    
    def diff_x(self, u, order):
        """Compute the spatial derivative of the 3D field u of order 'order' in x direction.
        """
        
        u_tilde = self.FFT_1D(u, axis=0)
        factor = (self._deriv_factor_x ** order).reshape(-1, 1, 1)
        return self.IFFT_1D(u_tilde * factor, axis=0)
    
    # def diff_x_state(self, q_vec, order):
    #     """Compute the spatial derivative of the FOM state q_vec (2*nx*ny*nz, ) order 'order' in x direction.
    #     """
    #     v = q_vec[:self.nx * self.ny * self.nz].reshape(self.nx, self.ny, self.nz)
    #     eta = q_vec[self.nx * self.ny * self.nz:].reshape(self.nx, self.ny, self.nz)
        
    #     dv_dx = self.diff_x(v, order)
    #     deta_dx = self.diff_x(eta, order)
        
    #     return np.concatenate((dv_dx.ravel(), deta_dx.ravel()))
    
    def diff_z(self, u, order):
        """Compute the spatial derivative of the 3D field u(x, y, z) of order 'order' in z direction.
        """
        
        u_tilde = self.FFT_1D(u, axis=2)
        factor = (self._deriv_factor_z ** order).reshape(1, 1, -1)
        return self.IFFT_1D(u_tilde * factor, axis=2)
    
    def diff_1_y(self, u):
        """Compute the first spatial derivative of the 3D field u(x, y, z) in y direction using Chebyshev differentiation matrix.
        """
        
        u_dy = np.einsum('mj, kjl -> kml', self.D1, u)
        return u_dy
    
    def diff_2_y(self, u):
        """Compute the second spatial derivative of the 3D field u(x, y, z) in y direction using Chebyshev differentiation matrix.
        """
        
        u_dyy = np.einsum('mj, kjl -> kml', self.D2, u)
        return u_dyy
    
    def diff_4_y(self, u):
        """Compute the fourth spatial derivative of the 3D field u(x, y, z) in y direction using Chebyshev differentiation matrix.
        """
        
        u_dyyy = np.einsum('mj, kjl -> kml', self.D4, u)
        return u_dyyy
    
    def inner_product_3D(self, v1, eta1, v2, eta2):
        
        """Compute the inner product of two system's states (v1, eta1) and (v2, eta2).
        
        Input: 
        v1 -- spatial function with shape (nx, ny, nz)
        eta1 -- spatial function with shape (nx, ny, nz)
        v2 -- spatial function with shape (nx, ny, nz)
        eta2 -- spatial function with shape (nx, ny, nz)
        
        Output: the disturbance kinetic energy
        
        sum_{k = -Nx/2}^{Nx/2 - 1} sum_{m = -Nz/2}^{Nz/2 - 1}
        1/(2 alpha^2 + 2 beta^2) * integral_{-1}^{1}  [dv1_ff_km/dy conj(dv2_ff_km/dy) + (alpha^2 + beta^2) v1_ff_km conj(v2_ff_km) + eta1_ff_km conj(eta2_ff_km)] dy
        
        """
        
        v1_breve = self.FFT_2D(v1)
        eta1_breve = self.FFT_2D(eta1)
        v2_breve = self.FFT_2D(v2)
        eta2_breve = self.FFT_2D(eta2)
        
        dv1_dy_breve = np.einsum('mj, kjl -> kml', self.D1, v1_breve)
        dv2_dy_breve = np.einsum('mj, kjl -> kml', self.D1, v2_breve)

        term1 = dv1_dy_breve * dv2_dy_breve.conj()
        term2 = self.k_sq * v1_breve * v2_breve.conj()
        term3 = eta1_breve * eta2_breve.conj()
        
        total_integrand = term1 + term2 + term3
        
        w = self.Clenshaw_Curtis_weights[None, :, None]
        integral_y = np.sum(total_integrand * w, axis=1)  # shape: (nx, nz)
        final_result = np.sum(integral_y * self.inner_product_factor)

        return np.real(final_result)
    
    def compute_snapshot_correlation_matrix(self, q_vec_snapshots):
        """
        Efficiently compute the correlation matrix C for a batch of snapshots using vectorized operations.
        
        Input:
            q_vec_snapshots -- vectorized spatial functions with shape (2 * nx * ny * nz, M_snapshots)
            
        Output:
            C -- Correlation matrix with shape (M_snapshots, M_snapshots)
                 where C[i, j] = inner_product_3D(snapshot[i], snapshot[j])
                 
        ## verified, this function produces identical results as the double loop version to compute the C matrix C_ij = <snapshot[i], snapshot[j]>_{inner product 3D} but is much faster.
        """
        
        # 1. Shape & Dimensions
        num_grid = self.nx * self.ny * self.nz
        M = q_vec_snapshots.shape[1]
        nx, ny, nz = self.nx, self.ny, self.nz
        
        # 2. Reshape and Split into v and eta (Batch mode)
        # Input shape: (2*grid, M) -> Transpose to (M, 2*grid) -> Reshape to (M, nx, ny, nz)
        snapshots_v = q_vec_snapshots[:num_grid, :].T.reshape(M, nx, ny, nz)
        snapshots_eta = q_vec_snapshots[num_grid:, :].T.reshape(M, nx, ny, nz)
        
        # 3. Batch FFT (Vectorized FFT_2D logic)
        # Apply FFT along x (axis 1) and z (axis 3) for the whole batch
        v_hat = np.fft.fft2(snapshots_v, axes=(1, 3))
        v_hat = np.fft.fftshift(v_hat, axes=(1, 3))
        v_hat /= (nx * nz) # Scaling consistent with FFT_2D
        
        eta_hat = np.fft.fft2(snapshots_eta, axes=(1, 3))
        eta_hat = np.fft.fftshift(eta_hat, axes=(1, 3))
        eta_hat /= (nx * nz) # Scaling consistent with FFT_2D
        
        # Nyquist Mode Handling (Vectorized)
        if (nx % 2) == 0:
            v_hat[:, 0, :, :] = 0
            eta_hat[:, 0, :, :] = 0
        if (nz % 2) == 0:
            v_hat[:, :, :, 0] = 0
            eta_hat[:, :, :, 0] = 0
            
        # 4. Batch Derivatives (Vectorized Chebyshev D1)
        # self.D1 is (ny, ny). We apply it to axis 2 (y) of the batch array (M, nx, ny, nz)
        # Einstein summation: D1[j,k] * v_hat[m,i,k,l] -> result[m,i,j,l]
        dv_dy_hat = np.einsum('jk, mikl -> mijl', self.D1, v_hat)
        
        # 5. Prepare Weights and Factors (Broadcasting)
        # Reshape weights to (1, nx, ny, nz) compatible shapes for broadcasting
        
        # Clenshaw-Curtis weights (y-direction): shape (ny,) -> (1, 1, ny, 1)
        w_expanded = self.Clenshaw_Curtis_weights.reshape(1, 1, ny, 1)
        
        # Inner Product Factor (x,z-direction): shape (nx, nz) -> (1, nx, 1, nz)
        # Note: self.inner_product_factor was squeezed in __init__
        factor_expanded = self.inner_product_factor.reshape(1, nx, 1, nz)
        
        # k_sq (x,z-direction): shape (nx, 1, nz) -> (1, nx, 1, nz)
        # Note: self.k_sq in __init__ is (nx, 1, nz), it broadcasts fine, but reshape is safer
        k_sq_expanded = self.k_sq.reshape(1, nx, 1, nz)
        
        # Combined Scaling Factor S = w * factor
        S = w_expanded * factor_expanded # shape (1, nx, ny, nz)
        sqrt_S = np.sqrt(S)
        
        # 6. Construct Feature Components (A * conj(B) -> A . B vector form)
        
        # Term 1: dv/dy * conj(dv/dy) -> Feature: dv/dy * sqrt(S)
        feat_1 = dv_dy_hat * sqrt_S
        
        # Term 2: k^2 * v * conj(v) -> Feature: v * sqrt(k^2) * sqrt(S)
        feat_2 = v_hat * np.sqrt(k_sq_expanded) * sqrt_S
        
        # Term 3: eta * conj(eta) -> Feature: eta * sqrt(S)
        feat_3 = eta_hat * sqrt_S
        
        # 7. Stack Features and Compute Matrix Product
        # Flatten the spatial dimensions: (M, total_features)
        Y = np.concatenate([
            feat_1.reshape(M, -1),
            feat_2.reshape(M, -1),
            feat_3.reshape(M, -1)
        ], axis=1)
        
        # Compute Correlation Matrix: C = Y @ Y.conj().T
        # This effectively sums over all spatial points (dot product)
        C = Y @ Y.conj().T
        
        return np.real(C)
    
    def Cholesky_inner_product_weight(self):
        """
        Precompute the Cholesky factors for the inner product weight operator W, such that W = R^T R, where W is 2N-by-2N symmetric positive definite, and R is a 2N-by-2N invertible matrix.
        """
        # print("Pre-computing spectral Cholesky factors (R matrices)...")
        
        # 初始化存储容器
        # R_blocks_v: (nx, nz, ny, ny) 
        # 保持不变，这是矩阵运算库，通常把 batch 放在前面 (nx, nz) 比较方便索引
        self.R_blocks_v = np.zeros((self.nx, self.nz, self.ny, self.ny), dtype=np.float64)
        
        # [修改点 1] R_diag_eta: 改为 (nx, ny, nz) 以匹配物理网格布局，方便直接广播
        self.R_diag_eta = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.inv_R_blocks_v = np.zeros((self.nx, self.nz, self.ny, self.ny), dtype=np.float64)
        self.inv_R_diag_eta = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        
        w_y = self.Clenshaw_Curtis_weights.flatten()
        fact_xz = self.inner_product_factor.reshape(self.nx, self.nz)
        k2_xz = self.k_sq.reshape(self.nx, self.nz)
        
        for i in range(self.nx):
            for k in range(self.nz):
                # 1. 构造当前模态的权重矩阵 S (对角)
                S_val = w_y * fact_xz[i, k]
                S_mat = np.diag(S_val)
                
                # 2. 构造 v 分量的能量矩阵 M_v = D^T S D + k^2 S
                # 这是 v 分量对 Energy Norm 的贡献核
                k2 = k2_xz[i, k]
                M_v = self.D1.T @ S_mat @ self.D1 + k2 * S_mat
                
                # 3. Cholesky 分解 M_v = R^T R
                # scipy返回的默认是上三角 R
                # 加一点 jitter 防止数值误差导致的非正定
                try:
                    R = scipy.linalg.cholesky(M_v + 1e-12 * np.eye(self.ny), lower=False)
                    R_inv = scipy.linalg.inv(R)
                except np.linalg.LinAlgError:
                    R = np.zeros((self.ny, self.ny)) # 处理奇异模态
                    R_inv = np.zeros((self.ny, self.ny))
                
                self.R_blocks_v[i, k, :, :] = R
                self.inv_R_blocks_v[i, k, :, :] = R_inv
                
                # [修改点 2] Eta 分量赋值: 也就是 sqrt(S)
                val = np.sqrt(S_val)
                self.R_diag_eta[i, :, k] = val
                self.inv_R_diag_eta[i, :, k] = np.divide(1.0, val, out=np.zeros_like(val), where=val!=0)

        # Nyquist 模态置零 (保持和你原逻辑一致)
        if (self.nx % 2) == 0: 
            self.R_blocks_v[0, :, :, :] = 0
            self.R_diag_eta[0, :, :] = 0 # x=0, all y, all z
            self.inv_R_blocks_v[0, :, :, :] = 0
            self.inv_R_diag_eta[0, :, :] = 0
            
        if (self.nz % 2) == 0: 
            self.R_blocks_v[:, 0, :, :] = 0
            self.R_diag_eta[:, :, 0] = 0
            self.inv_R_blocks_v[:, 0, :, :] = 0
            self.inv_R_diag_eta[:, :, 0] = 0
        
    def apply_sqrt_inner_product_weight(self, q_vec):
        """
        计算 u = R * q
        物理意义：变换后的向量 u，满足 ||u||_2^2 = ||q||_W^2
        Input: q_vec (2N, M)
        Output: q_R (2N, M)
        """
        # 1. 输入处理
        is_1d = (q_vec.ndim == 1)
        if is_1d: q_vec = q_vec[:, np.newaxis]
        M = q_vec.shape[1]
        
        num_grid = self.nx * self.ny * self.nz
        
        # 2. 变换到谱空间
        q_v = q_vec[:num_grid, :].T.reshape(M, self.nx, self.ny, self.nz)
        q_eta = q_vec[num_grid:, :].T.reshape(M, self.nx, self.ny, self.nz)
        
        # FFT (注意归一化，保持和你原来一致)
        v_hat = np.fft.fftshift(np.fft.fft2(q_v, axes=(1, 3)), axes=(1, 3)) / np.sqrt((self.nx * self.nz)) 
        eta_hat = np.fft.fftshift(np.fft.fft2(q_eta, axes=(1, 3)), axes=(1, 3)) / np.sqrt((self.nx * self.nz))
        
        # 3. 应用 R 矩阵 (Block-wise Multiplication)
        # v 分量: 矩阵乘法 R (ny, ny) * v (ny)
        # 调整维度以适配 einsum: M, nx, nz, ny (将 y 放到最后)
        v_hat_perm = v_hat.transpose(0, 1, 3, 2) 
        
        # self.R_blocks_v shape: (nx, nz, ny, ny) -> (ikab)
        # v_hat_perm shape: (M, nx, nz, ny) -> (mikb)
        # result shape: (M, nx, nz, ny) -> (mika)
        v_R_hat = np.einsum('ikab, mikb -> mika', self.R_blocks_v, v_hat_perm)
        v_R_hat = v_R_hat.transpose(0, 1, 3, 2) # 换回 (M, nx, ny, nz)
        
        # eta 分量: 对角乘法 (Element-wise)
        eta_R_hat = eta_hat * self.R_diag_eta[np.newaxis, :, :, :]
        
        # 4. 变回物理空间 (IFFT)
        # 理论上 R 是实算子，结果取实部即可
        q_v_out = np.fft.ifft2(np.fft.ifftshift(v_R_hat, axes=(1,3)), axes=(1,3)).real
        q_eta_out = np.fft.ifft2(np.fft.ifftshift(eta_R_hat, axes=(1,3)), axes=(1,3)).real
        
        # 5. 拼接
        res = np.concatenate([q_v_out.reshape(M, -1), q_eta_out.reshape(M, -1)], axis=1).T
        return res.squeeze() if is_1d else res

    def apply_sqrt_inner_product_weight_transpose(self, q_vec):
        """
        计算 w = R^T * q
        通常用于组合计算 W * q = R^T * (R * q)
        Input: q_vec (2N, M) -- 这通常是 _apply_R 的输出
        Output: q_out (2N, M)
        """
        # 1. 输入处理
        is_1d = (q_vec.ndim == 1)
        if is_1d: q_vec = q_vec[:, np.newaxis]
        M = q_vec.shape[1]
        num_grid = self.nx * self.ny * self.nz

        # 2. 变换到谱空间
        q_v = q_vec[:num_grid, :].T.reshape(M, self.nx, self.ny, self.nz)
        q_eta = q_vec[num_grid:, :].T.reshape(M, self.nx, self.ny, self.nz)
        
        # FFT
        v_hat = np.fft.fftshift(np.fft.fft2(q_v, axes=(1, 3)), axes=(1, 3)) / np.sqrt((self.nx * self.nz))
        eta_hat = np.fft.fftshift(np.fft.fft2(q_eta, axes=(1, 3)), axes=(1, 3)) / np.sqrt((self.nx * self.nz))
        
        # 3. 应用 R^T 矩阵
        # v 分量: 矩阵乘法 R^T (ny, ny) * v (ny)
        v_hat_perm = v_hat.transpose(0, 1, 3, 2)
        
        # 注意这里 einsum 的下标变化：
        # R_blocks_v 是 (row, col) = (a, b) -> R_ab
        # 我们要计算 R^T * v -> sum_row (R_row,col * v_row) -> sum_a (R_ab * v_a)
        # ikab: R_ab
        # mika: v_a
        # -> mikb: result_b
        v_RT_hat = np.einsum('ikab, mika -> mikb', self.R_blocks_v, v_hat_perm)
        v_RT_hat = v_RT_hat.transpose(0, 1, 3, 2)
        
        # eta 分量: 对角阵转置还是自己
        eta_RT_hat = eta_hat * self.R_diag_eta[np.newaxis, :, :, :]
        
        # 4. 变回物理空间
        q_v_out = np.fft.ifft2(np.fft.ifftshift(v_RT_hat, axes=(1,3)), axes=(1,3)).real
        q_eta_out = np.fft.ifft2(np.fft.ifftshift(eta_RT_hat, axes=(1,3)), axes=(1,3)).real
        
        # 5. 拼接
        res = np.concatenate([q_v_out.reshape(M, -1), q_eta_out.reshape(M, -1)], axis=1).T
        return res.squeeze() if is_1d else res
    
    def apply_inner_product_weight(self, q_vec):
        """
        计算 W * Basis = R^T * (R * Basis)
        """
        # 1. v -> Rv
        Rv = self.apply_sqrt_inner_product_weight(q_vec)
        
        # 2. Rv -> R^T Rv
        Wv = self.apply_sqrt_inner_product_weight_transpose(Rv)
        
        return Wv
    
    def apply_inv_sqrt_inner_product_weight(self, q_vec):
        """
        计算 R^-1 * q_vec
        Input: q_vec (2N, M) -- 特征空间变量
        Output: (2N, M) -- 物理空间变量
        """
        is_1d = (q_vec.ndim == 1)
        if is_1d: q_vec = q_vec[:, np.newaxis]
        M = q_vec.shape[1]
        num_grid = self.nx * self.ny * self.nz
        
        # 1. 变换到谱空间
        q_v = q_vec[:num_grid, :].T.reshape(M, self.nx, self.ny, self.nz)
        q_eta = q_vec[num_grid:, :].T.reshape(M, self.nx, self.ny, self.nz)
        
        # [Critical] 逆变换的缩放: FFT * sqrt(N)
        # 以抵消前向变换中的 / sqrt(N)
        scale_inv = np.sqrt(self.nx * self.nz)
        
        v_hat = np.fft.fftshift(np.fft.fft2(q_v, axes=(1, 3)), axes=(1, 3)) * scale_inv
        eta_hat = np.fft.fftshift(np.fft.fft2(q_eta, axes=(1, 3)), axes=(1, 3)) * scale_inv
        
        # 2. 应用 R^-1 矩阵
        v_hat_perm = v_hat.transpose(0, 1, 3, 2)
        
        # 使用 inv_R_blocks_v
        # 这里的 einsum 和 apply_sqrt... 完全一样，只是换了矩阵
        v_hat = np.einsum('ikab, mikb -> mika', self.inv_R_blocks_v, v_hat_perm)
        v_hat = v_hat.transpose(0, 1, 3, 2)
        
        # 使用 inv_R_diag_eta
        eta_hat = eta_hat * self.inv_R_diag_eta[np.newaxis, :, :, :]
        
        # 3. IFFT 回物理空间
        q_v_out = np.fft.ifft2(np.fft.ifftshift(v_hat, axes=(1,3)), axes=(1,3)).real
        q_eta_out = np.fft.ifft2(np.fft.ifftshift(eta_hat, axes=(1,3)), axes=(1,3)).real
        
        res = np.concatenate([q_v_out.reshape(M, -1), q_eta_out.reshape(M, -1)], axis=1).T
        return res.squeeze() if is_1d else res
    
    def apply_inv_sqrt_inner_product_weight_transpose(self, q_vec):
        """
        计算 (R^-1)^T * q_vec
        通常用于对梯度进行预处理 (Preconditioning)
        """
        is_1d = (q_vec.ndim == 1)
        if is_1d: q_vec = q_vec[:, np.newaxis]
        M = q_vec.shape[1]
        num_grid = self.nx * self.ny * self.nz
        
        # 1. 变换到谱空间
        q_v = q_vec[:num_grid, :].T.reshape(M, self.nx, self.ny, self.nz)
        q_eta = q_vec[num_grid:, :].T.reshape(M, self.nx, self.ny, self.nz)
        
        # [Critical] 逆变换缩放
        scale_inv = np.sqrt(self.nx * self.nz)
        v_hat = np.fft.fftshift(np.fft.fft2(q_v, axes=(1, 3)), axes=(1, 3)) * scale_inv
        eta_hat = np.fft.fftshift(np.fft.fft2(q_eta, axes=(1, 3)), axes=(1, 3)) * scale_inv
        
        # 2. 应用 (R^-1)^T
        v_hat_perm = v_hat.transpose(0, 1, 3, 2)
        
        # 使用 inv_R_blocks_v，但是用转置的 einsum 索引
        # 'ikab, mika -> mikb' 表示对 inv_R 的第一个维度(行)求和，即左乘 inv_R.T
        v_hat = np.einsum('ikab, mika -> mikb', self.inv_R_blocks_v, v_hat_perm)
        v_hat = v_hat.transpose(0, 1, 3, 2)
        
        # 对角阵转置不变
        eta_hat = eta_hat * self.inv_R_diag_eta[np.newaxis, :, :, :]
        
        # 3. IFFT
        q_v_out = np.fft.ifft2(np.fft.ifftshift(v_hat, axes=(1,3)), axes=(1,3)).real
        q_eta_out = np.fft.ifft2(np.fft.ifftshift(eta_hat, axes=(1,3)), axes=(1,3)).real
        
        res = np.concatenate([q_v_out.reshape(M, -1), q_eta_out.reshape(M, -1)], axis=1).T
        return res.squeeze() if is_1d else res
    
    def apply_inv_inner_product_weight(self, q_vec):
        """
        计算 W^-1 * q_vec = (R^T R)^-1 * q_vec = R^-1 R^-T * q_vec
        """
        # 1. 计算 R^-T * q_vec
        R_T_inv_q = self.apply_inv_sqrt_inner_product_weight_transpose(q_vec)
        # 2. 计算 R^-1 * (R^-T * q_vec)
        W_inv_q = self.apply_inv_sqrt_inner_product_weight(R_T_inv_q)
        
        return W_inv_q

    def shift_x_input_3D(self, u, c):
        """Shift the 3D field u(x, y, z) by amount c in x direction to get u(x - c, y, z).
        """
        
        u_tilde = self.FFT_1D(u, axis=0)
        u_tilde_shifted = u_tilde * np.exp(2j * np.pi * self.mode_idx_x.reshape(-1, 1, 1) * (-c) / self.Lx)
        return self.IFFT_1D(u_tilde_shifted, axis=0)
    
    def shift_x_state(self, q_vec, c):
        """Shift the FOM state q_vec (2*nx*ny*nz, ) by amount c in x direction to get q_vec shifted.
        """
        v = q_vec[:self.nx * self.ny * self.nz].reshape(self.nx, self.ny, self.nz)
        eta = q_vec[self.nx * self.ny * self.nz:].reshape(self.nx, self.ny, self.nz)
        
        v_shifted = self.shift_x_input_3D(v, c)
        eta_shifted = self.shift_x_input_3D(eta, c)
        
        return np.concatenate((v_shifted.ravel(), eta_shifted.ravel()))

    def shift_z_input_3D(self, u, c):
        """Shift the 3D field u(x, y, z) by amount c in z direction to get u(x, y, z - c).
        """
        
        u_tilde = self.FFT_1D(u, axis=2)
        u_tilde_shifted = u_tilde * np.exp(2j * np.pi * self.mode_idx_z.reshape(1, 1, -1) * (-c) / self.Lz)
        return self.IFFT_1D(u_tilde_shifted, axis=2)
    
    def template_fitting(self, q_vec):
        """Perform template fitting to remove the translational symmetry in x and z directions.
        
        Input:
            q_vec -- vectorized spatial function with shape (2 * nx * ny * nz, n_snapshots)
            q_template_vec -- vectorized spatial function with shape (2 * nx * ny * nz, )
            
        Algorithm:
            We use the projection onto the first Fourier slice to determine the shifting amount
            
        Output:
            q_vec_fitted -- vectorized spatial function with shape (2 * nx * ny * nz, n_snapshots)
            shifting_amounts -- array of shape (n_snapshots), where the first column is the shifting amount in x direction (temporarily, we only consider x direction shifting)
        """
        
        N_snapshots = q_vec.shape[1]
        v_snapshots = q_vec[0 : self.nx * self.ny * self.nz, :].reshape((self.nx, self.ny, self.nz, N_snapshots))
        eta_snapshots = q_vec[self.nx * self.ny * self.nz : , :].reshape((self.nx, self.ny, self.nz, N_snapshots))
        
        shifting_amount = np.zeros(N_snapshots)
        v_snapshots_fitted = np.zeros_like(v_snapshots)
        eta_snapshots_fitted = np.zeros_like(eta_snapshots)
        
        for idx_snapshot in range(N_snapshots):
            v_snapshot = v_snapshots[:, :, :, idx_snapshot]
            eta_snapshot = eta_snapshots[:, :, :, idx_snapshot]
            
            inner_product_q_template = self.inner_product_3D(v_snapshot, eta_snapshot,
                                                            self.v_template, self.eta_template)
            
            inner_product_q_template_quarter_shifted = self.inner_product_3D(v_snapshot, eta_snapshot,
                                                                             self.v_template_x_quarter_shifted,
                                                                             self.eta_template_x_quarter_shifted)
            
            shifting_amount[idx_snapshot] = np.angle(inner_product_q_template + 1j * inner_product_q_template_quarter_shifted) * (self.Lx / (2 * np.pi))
            v_snapshots_fitted[:, :, :, idx_snapshot] = self.shift_x_input_3D(v_snapshot, -shifting_amount[idx_snapshot])
            eta_snapshots_fitted[:, :, :, idx_snapshot] = self.shift_x_input_3D(eta_snapshot, -shifting_amount[idx_snapshot])
            
        q_vec_fitted = np.concatenate((v_snapshots_fitted.reshape((self.nx * self.ny * self.nz, N_snapshots)),
                                      eta_snapshots_fitted.reshape((self.nx * self.ny * self.nz, N_snapshots))), axis=0)
        
        shifting_amount = np.unwrap(shifting_amount, period=self.Lx) # unwrap the shifting amount to avoid discontinuities due to periodicity
        
        return q_vec_fitted, shifting_amount
   
    def assemble_fom_linear_operator(self):
        """Assemble the full linear operator L for the FOM."""
        
        Id = np.eye(self.ny - 2, dtype=complex)
        D2 = self.D2
        D4 = self.D4
        
        M = np.zeros((2 * (self.ny - 2), 2 * (self.ny - 2)), dtype=complex)
        L = np.zeros((2 * (self.ny - 2), 2 * (self.ny - 2)), dtype=complex)
        linear_mat = np.zeros((self.nx, self.nz, 2 * (self.ny - 2), 2 * (self.ny - 2)), dtype=complex)
        
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
                linear_mat[idx_kx, idx_kz, :, :] = scipy.linalg.solve(M, L)
                
        return linear_mat
    
    def assemble_fom_exp_linear_operator(self, fom_linear_mat, dt):
        """Assemble the matrix exponential of the full linear operator exp(dt*L) for the exponential timestepper of the FOM."""
        
        exp_linear_mat = np.zeros((self.nx, self.nz, 2 * (self.ny - 2), 2 * (self.ny - 2)), dtype=complex)
        for idx_kx in range(self.nx):
            for idx_kz in range(self.nz):
                exp_linear_mat[idx_kx, idx_kz, :, :] = scipy.linalg.expm(fom_linear_mat[idx_kx, idx_kz, :, :] * dt)
        return exp_linear_mat
 
    def linear(self, q_vec):
        """Evaluate linear operator Lq.
        
        Input:
            q_vec -- vectorized spatial function with shape (2 * nx * ny * nz, )
        Output:
            Lq_vec -- vectorized spatial function with shape (2 * nx * ny * nz, )
        
        We apply the linear operator in 3 steps
        1. Transform via 2D Fourier transform in x and z directions
        2. For each (kx, kz) mode, apply the linear operator in y direction via differentiation matrices obtained from Chebyshev collocation method w/ BCs
        3. Transform back to physical space via inverse 2D Fourier transform in x and z directions
        """
        
        v = q_vec[0 : self.nx * self.ny * self.nz].reshape((self.nx, self.ny, self.nz))
        eta = q_vec[self.nx * self.ny * self.nz : ].reshape((self.nx, self.ny, self.nz))
        
        v_breve = self.FFT_2D(v)
        eta_breve = self.FFT_2D(eta)
        
        v_breve_inner = v_breve[:, 1:-1, :].transpose(0, 2, 1)   # (nx, nz, ny-2)
        eta_breve_inner = eta_breve[:, 1:-1, :].transpose(0, 2, 1) # (nx, nz, ny-2)
        state_breve = np.concatenate((v_breve_inner, eta_breve_inner), axis=2) # (nx, nz, 2*(ny-2))
        state_breve = state_breve[..., np.newaxis] # (nx, nz, 2*(ny-2)) -> (nx, nz, 2*(ny-2), 1)

        # self.linear_mat shape: (nx, nz, 2*(ny-2), 2*(ny-2))
        # state_breve     shape: (nx, nz, 2*(ny-2), 1)
        # out_breve       shape: (nx, nz, 2*(ny-2), 1)
        linear_state_breve_inner = self.linear_mat @ state_breve 
        linear_state_breve_inner = linear_state_breve_inner[..., 0] # (nx, nz, 2*(ny-2))
        linear_v_breve_inner = linear_state_breve_inner[:, :, :self.ny - 2]   # (nx, nz, ny-2)
        linear_eta_breve_inner = linear_state_breve_inner[:, :, self.ny - 2:] # (nx, nz, ny-2)

        linear_v_breve = np.zeros_like(v_breve)
        linear_eta_breve = np.zeros_like(eta_breve)        
        linear_v_breve[:, 1:-1, :] = linear_v_breve_inner.transpose(0, 2, 1) # (nx, nz, ny-2) -> (nx, ny-2, nz)
        linear_eta_breve[:, 1:-1, :] = linear_eta_breve_inner.transpose(0, 2, 1) # (nx, nz, ny-2) -> (nx, ny-2, nz)
        linear_v = self.IFFT_2D(linear_v_breve)
        linear_eta = self.IFFT_2D(linear_eta_breve)
        
        return np.concatenate((linear_v.ravel(), linear_eta.ravel()))
    
    def assemble_weighted_petrov_galerkin_tensors(self, Psi_w, PhiF_w):
        """Assemble the coefficients of reduced-order dynamics for the weighted mode amplitude Ra via Petrov-Galerkin projection.
        Notice that x-derivative is commutative with the weight matrix R and W = R^T R,
        however, the linear dynamics is not commutative with R in general because it involves y-derivatives and multiplication by base flow U(y).
        """
        
        r = Psi_w.shape[1]
        A_mat = np.zeros((r, r))
        p_vec = np.zeros(r)
        s_vec = np.zeros(r)
        
        R_T_Psi_w = self.apply_sqrt_inner_product_weight_transpose(Psi_w)
        R_inv_PhiF_w = self.apply_inv_sqrt_inner_product_weight(PhiF_w)

        PhiF_w_dx = self.diff_x_basis(PhiF_w, order=1)
        
        M_mat = Psi_w.T @ PhiF_w_dx
        
        for i in range(r):
            linear_R_inv_PhiF_w_i = self.linear(R_inv_PhiF_w[:, i])
            R_linear_R_inv_PhiF_w_i = self.apply_sqrt_inner_product_weight(linear_R_inv_PhiF_w_i)
            p_vec[i] = np.dot(R_linear_R_inv_PhiF_w_i, self.q_template_dx_vec_w)
            s_vec[i] = np.dot(PhiF_w_dx[:, i], self.q_template_dx_vec_w)
            for j in range(r):
                A_mat[i, j] = np.dot(R_T_Psi_w[:, i], self.linear(R_inv_PhiF_w[:, j]))

        return (A_mat, p_vec, s_vec, M_mat)
    
    def diff_x_basis(self, Phi, order):
        """Compute the spatial derivative of the basis functions in x direction."""
        
        r = Phi.shape[1]
        Phi_dx = np.zeros_like(Phi)
        
        for i in range(r):
            v_basis = Phi[0 : self.nx * self.ny * self.nz, i].reshape((self.nx, self.ny, self.nz))
            eta_basis = Phi[self.nx * self.ny * self.nz : , i].reshape((self.nx, self.ny, self.nz))
            v_basis_dx = self.diff_x(v_basis, order=order)
            eta_basis_dx = self.diff_x(eta_basis, order=order)
            Phi_dx[0 : self.nx * self.ny * self.nz, i] = v_basis_dx.ravel()
            Phi_dx[self.nx * self.ny * self.nz : , i] = eta_basis_dx.ravel()
        
        return Phi_dx
    
    def exp_linear_frequency_domain(self, exp_linear_mat, q_breve_vec):
        """Evaluate exp-linear operator exp(dt*L)q for the exponential timestepper."""
        
        v_breve = q_breve_vec[0 : self.nx * self.ny * self.nz].reshape((self.nx, self.ny, self.nz))
        eta_breve = q_breve_vec[self.nx * self.ny * self.nz : ].reshape((self.nx, self.ny, self.nz))
        
        v_breve_inner = v_breve[:, 1:-1, :].transpose(0, 2, 1)   # (nx, nz, ny-2)
        eta_breve_inner = eta_breve[:, 1:-1, :].transpose(0, 2, 1) # (nx, nz, ny-2)
        state_breve = np.concatenate((v_breve_inner, eta_breve_inner), axis=2) # (nx, nz, 2*(ny-2))
        state_breve = state_breve[..., np.newaxis] # (nx, nz, 2*(ny-2)) -> (nx, nz, 2*(ny-2), 1)

        # self.linear_mat shape: (nx, nz, 2*(ny-2), 2*(ny-2))
        # state_breve     shape: (nx, nz, 2*(ny-2), 1)
        # out_breve       shape: (nx, nz, 2*(ny-2), 1)
        linear_state_breve_inner = exp_linear_mat @ state_breve 
        linear_state_breve_inner = linear_state_breve_inner[..., 0] # (nx, nz, 2*(ny-2))
        linear_v_breve_inner = linear_state_breve_inner[:, :, :self.ny - 2]   # (nx, nz, ny-2)
        linear_eta_breve_inner = linear_state_breve_inner[:, :, self.ny - 2:] # (nx, nz, ny-2)

        linear_v_breve = np.zeros_like(v_breve)
        linear_eta_breve = np.zeros_like(eta_breve)        
        linear_v_breve[:, 1:-1, :] = linear_v_breve_inner.transpose(0, 2, 1) # (nx, nz, ny-2) -> (nx, ny-2, nz)
        linear_eta_breve[:, 1:-1, :] = linear_eta_breve_inner.transpose(0, 2, 1) # (nx, nz, ny-2) -> (nx, ny-2, nz)
        
        return np.concatenate((linear_v_breve.ravel(), linear_eta_breve.ravel()))
    
    def evaluate_fom_rhs_unreduced(self, q_vec):
        """Evaluate the FOM RHS for 3D velocity field."""
        return self.linear(q_vec)
    
    def load_template(self, q_template_vec, q_template_dx_vec):
        """Load the template function and its derivative for template fitting."""
        self.v_template_vec      = q_template_vec[0 : self.nx * self.ny * self.nz]
        self.v_template          = self.v_template_vec.reshape((self.nx, self.ny, self.nz))
        self.v_template_x_quarter_shifted = self.shift_x_input_3D(self.v_template, self.Lx / 4)
        self.v_template_dx_vec   = q_template_dx_vec[0 : self.nx * self.ny * self.nz]
        self.v_template_dx       = self.v_template_dx_vec.reshape((self.nx, self.ny, self.nz))
        
        self.eta_template_vec    = q_template_vec[self.nx * self.ny * self.nz : ]
        self.eta_template        = self.eta_template_vec.reshape((self.nx, self.ny, self.nz))
        self.eta_template_x_quarter_shifted = self.shift_x_input_3D(self.eta_template, self.Lx / 4)
        self.eta_template_dx_vec = q_template_dx_vec[self.nx * self.ny * self.nz : ]
        self.eta_template_dx     = self.eta_template_dx_vec.reshape((self.nx, self.ny, self.nz))
        
        self.q_template_vec_w = self.apply_sqrt_inner_product_weight(q_template_vec)
        self.q_template_dx_vec_w = self.apply_sqrt_inner_product_weight(q_template_dx_vec)
        self.v_template_dx_w = self.q_template_dx_vec_w[0 : self.nx * self.ny * self.nz].reshape((self.nx, self.ny, self.nz))
        self.eta_template_dx_w = self.q_template_dx_vec_w[self.nx * self.ny * self.nz : ].reshape((self.nx, self.ny, self.nz))    
    
    def evaluate_fom_shifting_speed_denom(self, q_fitted_vec):
        """Evaluate the shift speed for the FOM."""
        v_fitted = q_fitted_vec[0 : self.nx * self.ny * self.nz].reshape((self.nx, self.ny, self.nz))
        eta_fitted = q_fitted_vec[self.nx * self.ny * self.nz : ].reshape((self.nx, self.ny, self.nz))
        v_fitted_dx = self.diff_x(v_fitted, order=1)
        eta_fitted_dx = self.diff_x(eta_fitted, order=1)
        return self.inner_product_3D(v_fitted_dx,
                                     eta_fitted_dx,
                                     self.v_template_dx,
                                     self.eta_template_dx)
        
    def evaluate_fom_shifting_speed_numer(self, dqdt_unreduced_fitted_vec):
        """Evaluate the shift speed for the FOM."""
        dvdt_original_fitted = dqdt_unreduced_fitted_vec[0 : self.nx * self.ny * self.nz].reshape((self.nx, self.ny, self.nz))
        detadt_original_fitted = dqdt_unreduced_fitted_vec[self.nx * self.ny * self.nz : ].reshape((self.nx, self.ny, self.nz))
        return -self.inner_product_3D(dvdt_original_fitted,
                                     detadt_original_fitted,
                                     self.v_template_dx,
                                     self.eta_template_dx)
        
    def evaluate_fom_shifting_speed(self, q_fitted_vec, dqdt_unreduced_fitted_vec):
        """Evaluate the shift speed for the FOM."""
        return self.evaluate_fom_shifting_speed_numer(dqdt_unreduced_fitted_vec) / self.evaluate_fom_shifting_speed_denom(q_fitted_vec)
  
class time_step_LNS:
    
    def __init__(self, fom, time):
        self.fom = fom
        self.time = time
        self.dt = time[1] - time[0]
        self.nsteps = len(time)
        self.exp_linear_mat = fom.assemble_fom_exp_linear_operator(fom.linear_mat, self.dt)
  
    def time_step(self, q0_vec, nsave, *argv):

        """Time-step the KSE from initial condition q0 using CNRK3 scheme.
        See Peyret (2002) "Spectral Methods for Incompressible Viscous Flow" for details.
        
        args:
            q0_vec: initial condition (2 * nx * ny * nz 1D array, the first nx * ny * nz entries are v, the next nx * ny * nz entries are eta)
            nsave: save every nsave time steps (int)
            *argv: additional arguments for the forcing function (if any)
        returns:
            Qkj: state snapshots (Nx by Nt/nsave array)
            tsave: saved time instances (1D array)
        """
        tsave = self.time[::nsave]
        q_vec_snapshots = np.zeros((2 * self.fom.nx * self.fom.ny * self.fom.nz, len(tsave)))
        
        v = q0_vec[0 : self.fom.nx * self.fom.ny * self.fom.nz].reshape((self.fom.nx, self.fom.ny, self.fom.nz))
        eta = q0_vec[self.fom.nx * self.fom.ny * self.fom.nz : ].reshape((self.fom.nx, self.fom.ny, self.fom.nz))
        v_breve = self.fom.FFT_2D(v)
        eta_breve = self.fom.FFT_2D(eta)
        q_breve_vec = np.concatenate((v_breve.ravel(), eta_breve.ravel()))
 
        for idx_time in range(self.nsteps):
            t_current = idx_time * self.dt
            if idx_time % nsave == 0:
                print(f"Processing t={t_current:.3f}")
                q_vec_snapshots[:self.fom.nx * self.fom.ny * self.fom.nz, idx_time // nsave] = self.fom.IFFT_2D(q_breve_vec[:self.fom.nx * self.fom.ny * self.fom.nz].reshape((self.fom.nx, self.fom.ny, self.fom.nz))).ravel()
                q_vec_snapshots[self.fom.nx * self.fom.ny * self.fom.nz:, idx_time // nsave] = self.fom.IFFT_2D(q_breve_vec[self.fom.nx * self.fom.ny * self.fom.nz:].reshape((self.fom.nx, self.fom.ny, self.fom.nz))).ravel()
            q_breve_vec = self.fom.exp_linear_frequency_domain(self.exp_linear_mat, q_breve_vec)
            
        return q_vec_snapshots, tsave