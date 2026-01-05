import numpy as np 
import scipy as sp
import math
from string import ascii_lowercase as ascii
from mpi4py import MPI
import matplotlib.pyplot as plt

def compute_indices(c_ls=[], c=0, idx=5, r=5, order=0):
    
    for i in range(idx):
        ci = c + i * r**order
        if (order == 0):    c_ls.append(ci)
        else:               c_ls = compute_indices(c_ls,ci,i+1,r,order-1)
    return c_ls

# def perform_POD(pool,opt_obj,r):

#     X = np.ascontiguousarray(opt_obj.X_fitted, dtype=np.double)

#     N_space = X.shape[1]
#     N_snapshots = opt_obj.n_snapshots

#     if pool.rank == 0:
#         X_all = np.empty((pool.n_traj, N_space, N_snapshots))
#     else:
#         X_all = None

#     my_counts = pool.counts * N_space * N_snapshots
#     my_disps = pool.disps * N_space * N_snapshots
    
#     pool.comm.Gatherv(sendbuf = X,
#                       recvbuf = [X_all, my_counts, my_disps, MPI.DOUBLE], root=0)
    
#     if pool.rank == 0:
#         X_all = X_all.transpose(1,0,2).reshape(N_space,-1)
#         # plt.figure(figsize=(10,6))
#         # plt.contourf(X_all.T)
#         # plt.colorbar()
#         # plt.xlabel(r"$x$")
#         # plt.ylabel(r"$t$")
#         # plt.tight_layout()
#         # plt.show()
#         U, S, _ = sp.linalg.svd(X_all, full_matrices=False)
#         Phi = np.ascontiguousarray(U[:,:r])
#         cumulative_energy_proportion = 100 * np.cumsum(S[:r]**2) / np.sum(S**2)
#     else:
#         Phi = np.empty((N_space,r))
#         cumulative_energy_proportion = np.empty(r)
    
#     pool.comm.Bcast(Phi, root=0)
#     pool.comm.Bcast(cumulative_energy_proportion, root=0)
#     pool.comm.Barrier()
    
#     if pool.rank == 0:
#         print('POD complete on the primary processor and distributed, energy captured: %.4f%%'%(cumulative_energy_proportion[-1]))
    
#     return Phi, cumulative_energy_proportion

# def perform_POD_snapshot(pool, opt_obj, r, fom):

#     X = np.ascontiguousarray(opt_obj.X_fitted, dtype=np.double)
#     N_space = X.shape[1]
#     N_snapshots = opt_obj.n_snapshots

#     if pool.rank == 0:
#         X_all = np.empty((pool.n_traj, N_space, N_snapshots))
#     else:
#         X_all = None

#     my_counts = pool.counts * N_space * N_snapshots
#     my_disps = pool.disps * N_space * N_snapshots
    
#     # 收集所有快照到 rank 0
#     pool.comm.Gatherv(sendbuf = X,
#                       recvbuf = [X_all, my_counts, my_disps, MPI.DOUBLE], root=0)
    
#     if pool.rank == 0:
#         # Reshape: (N_space, Total_Snapshots)
#         # X_all 的列向量就是每一个时刻的快照 q(t)
#         X_all = X_all.transpose(1,0,2).reshape(N_space,-1)
#         M_total = X_all.shape[1] # 总快照数
        
#         # ==========================================================
#         # Method of Snapshots
#         # ==========================================================
        
#         # 2. 构建时间相关矩阵 C (M x M)
#         print(f"Rank 0: Building Correlation Matrix ({M_total}x{M_total}) using fom.inner_product_3D...")
#         C = np.zeros((M_total, M_total))
        
#         for i in range(M_total):
#             # 提取第 i 个快照 (扁平化向量)
#             q1 = X_all[:, i] 
            
#             # 注意：如果 fom.inner_product_3D 严格需要 (nx, ny, nz) 形状的输入
#             # 你可能需要在这里 reshape，例如: q1 = q1.reshape(nx, ny, nz, order='F')
#             # 这里假设它能接受扁平向量或者在内部处理了
            
#             for j in range(i, M_total):
#                 print(f"Computing C[{i},{j}]...")
#                 q2 = X_all[:, j]
                
#                 # 调用你的自定义加权内积函数
#                 val = fom.inner_product_3D(q1[:fom.nx * fom.ny * fom.nz].reshape((fom.nx, fom.ny, fom.nz)),
#                                            q1[fom.nx * fom.ny * fom.nz:].reshape((fom.nx, fom.ny, fom.nz)),
#                                            q2[:fom.nx * fom.ny * fom.nz].reshape((fom.nx, fom.ny, fom.nz)),
#                                            q2[fom.nx * fom.ny * fom.nz:].reshape((fom.nx, fom.ny, fom.nz)))
                
#                 C[i, j] = val
#                 C[j, i] = val # 利用对称性

#         # 3. 求解特征值问题 C * v = lambda * v
#         # eigh 专门用于实对称矩阵，速度快且更稳
#         eigvals, eigvecs = sp.linalg.eigh(C)
        
#         # 4. 排序 (eigh 返回的是从小到大，需要反转)
#         sorted_indices = np.argsort(eigvals)[::-1]
#         lambdas = eigvals[sorted_indices]
#         V = eigvecs[:, sorted_indices] # V 的每一列是特征向量 v_k
        
#         # 5. 截断 (只取前 r 个)
#         # 很多时候 lambda 会有极小的机器误差负数，取 abs 或者截断
#         lambdas_r = lambdas[:r]
#         V_r = V[:, :r]
        
#         # 计算能量占比 (基于特征值 lambda)
#         # 在快照法中，lambda 本身就是能量 (Sigma^2)
#         cumulative_energy_proportion = 100 * np.cumsum(lambdas[:r]) / np.sum(lambdas)
        
#         # 6. 重构空间模态 Phi
#         # Phi = X * V * S^(-1/2)
#         # 归一化因子: 1 / sqrt(lambda)
#         print("Rank 0: Reconstructing Spatial POD modes...")
        
#         inv_sqrt_S = np.diag(1.0 / np.sqrt(lambdas_r))
        
#         # 矩阵乘法重构: (N_space, M) @ (M, r) @ (r, r)
#         Phi = X_all @ V_r @ inv_sqrt_S
        
#         # 确保数据连续性，防止 MPI 报错
#         Phi = np.ascontiguousarray(Phi)
        
#     else:
#         Phi = np.empty((N_space,r))
#         cumulative_energy_proportion = np.empty(r)
    
#     # 分发结果 (和原来一样)
#     pool.comm.Bcast(Phi, root=0)
#     pool.comm.Bcast(cumulative_energy_proportion, root=0)
#     pool.comm.Barrier()
    
#     if pool.rank == 0:
#         print('POD (Snapshot Method) complete. Energy captured: %.4f%%'%(cumulative_energy_proportion[-1]))
    
#     return Phi, cumulative_energy_proportion

def perform_POD(pool, opt_obj, r, fom):
    
    """
    Verified. The POD modes are orthogonal under the weighted inner product defined in fom.inner_product_3D.
    """

    X = np.ascontiguousarray(opt_obj.X_fitted, dtype=np.double)
    N_space = X.shape[1]
    N_snapshots = opt_obj.n_snapshots

    if pool.rank == 0:
        X_all = np.empty((pool.n_traj, N_space, N_snapshots))
    else:
        X_all = None

    my_counts = pool.counts * N_space * N_snapshots
    my_disps = pool.disps * N_space * N_snapshots
    
    # 收集所有快照到 rank 0
    pool.comm.Gatherv(sendbuf = X,
                      recvbuf = [X_all, my_counts, my_disps, MPI.DOUBLE], root=0)
    
    if pool.rank == 0:
        # Reshape: (N_space, Total_Snapshots)
        X_all = X_all.transpose(1,0,2).reshape(N_space,-1)
        M_total = X_all.shape[1]
        
        C = fom.compute_snapshot_correlation_matrix(X_all) 
        
        print(f"Rank 0: Correlation Matrix computed. Range: {C.min():.2e} to {C.max():.2e}")

        # 3. 求解特征值问题 C * v = lambda * v
        # eigh 专门用于实对称矩阵，速度快且更稳
        eigvals, eigvecs = sp.linalg.eigh(C)
        
        # 4. 排序 (eigh 返回的是从小到大，需要反转)
        sorted_indices = np.argsort(eigvals)[::-1]
        lambdas = eigvals[sorted_indices]
        V = eigvecs[:, sorted_indices] # V 的每一列是特征向量 v_k
        
        # 5. 截断 (只取前 r 个)
        # 很多时候 lambda 会有极小的机器误差负数，取 abs 或者截断
        lambdas_r = lambdas[:r]
        V_r = V[:, :r]
        
        if np.any(lambdas_r <= 0):
            print(f"Rank 0 Warning: Detected {np.sum(lambdas_r <= 0)} non-positive eigenvalues due to numerical noise.")
            
            # 方法 A：取绝对值 (推荐，简单粗暴处理 -1e-16 这种噪声)
            lambdas_r = np.abs(lambdas_r)
            
            # 方法 B（更稳健）：防止除以极小值导致模态爆炸
            # 如果特征值太小（接近机器精度），它的倒数会极大，放大噪声。
            # 这里设置一个安全阈值 epsilon
            epsilon = 1e-14
            
            # 如果特征值小于 epsilon，说明该阶模态不仅是噪声，而且计算其逆平方根会引入巨大误差
            # 这种情况下，通常建议减小 r。但为了让程序不崩，我们可以将其“截断”在 epsilon
            if np.any(lambdas_r < epsilon):
                print(f"Rank 0 Warning: Some eigenvalues are smaller than {epsilon}. Clipping them to avoid division by zero.")
                lambdas_r = np.maximum(lambdas_r, epsilon)
        
        
        # 计算能量占比 (基于特征值 lambda)
        # 在快照法中，lambda 本身就是能量 (Sigma^2)
        cumulative_energy_proportion = 100 * np.cumsum(lambdas[:r]) / np.sum(lambdas)
        
        # 6. 重构空间模态 Phi
        # Phi = X * V * S^(-1/2)
        # 归一化因子: 1 / sqrt(lambda)
        print("Rank 0: Reconstructing Spatial POD modes...")
        
        inv_sqrt_S = np.diag(1.0 / np.sqrt(lambdas_r))
        
        # 矩阵乘法重构: (N_space, M) @ (M, r) @ (r, r)
        Phi = X_all @ V_r @ inv_sqrt_S
        
        # 确保数据连续性，防止 MPI 报错
        Phi = np.ascontiguousarray(Phi)
        
    else:
        Phi = np.empty((N_space,r))
        cumulative_energy_proportion = np.empty(r)
    
    # 分发结果 (和原来一样)
    pool.comm.Bcast(Phi, root=0)
    pool.comm.Bcast(cumulative_energy_proportion, root=0)
    pool.comm.Barrier()
    
    if pool.rank == 0:
        print('POD (Snapshot Method) complete. Energy captured: %.4f%%'%(cumulative_energy_proportion[-1]))
    
    return Phi, cumulative_energy_proportion

def assemble_Y(mpi_pool,Phi):
    
    r = Phi.shape[-1]
    Y = np.zeros((r,mpi_pool.n_traj*mpi_pool.n_snapshots))
    for i in range (mpi_pool.n_traj):
        Y[:,i*mpi_pool.n_snapshots:(i+1)*mpi_pool.n_snapshots] = Phi.T@mpi_pool.dX[i,]
    
    return Y

def assemble_W(mpi_pool):
    
    W = np.zeros(mpi_pool.n_traj*mpi_pool.n_snapshots)
    for i in range (mpi_pool.n_traj):
        W[i*mpi_pool.n_snapshots:(i+1)*mpi_pool.n_snapshots] = 1./(mpi_pool.weights[i]*mpi_pool.n_traj)
    
    return np.diag(W)


def assemble_Z(mpi_pool,Phi,poly_comp):
    
    r = Phi.shape[-1]
    
    for (count,p) in enumerate(poly_comp):
        
        rp = math.comb(r+p-1,p)
        idces = compute_indices([],0,r,r,p-1)
        equation = ','.join(ascii[:p])
        Z_ = np.zeros((rp,mpi_pool.n_traj*mpi_pool.n_snapshots))
        for i in range (mpi_pool.n_traj):
            for j in range (mpi_pool.n_snapshots):
                idx = i*mpi_pool.n_snapshots + j
                operands = [Phi.T@mpi_pool.X[i,:,j] for _ in range (p)]
                Z_[:,idx] = (np.einsum(equation,*operands).reshape(-1))[idces]
                
        if count == 0:  Z = Z_.copy()
        else:           Z = np.concatenate((Z,Z_),axis=0)


    return Z

def assemble_P(r,poly_comp,lambdas):
    
    for (count,p) in enumerate(poly_comp):
        
        rp = math.comb(r+p-1,p)
        P_ = lambdas[count]*np.ones(rp)
        
        if count == 0:  P = P_.copy()
        else:           P = np.concatenate((P,P_))
    
    return np.diag(P)


def extract_tensors(r,poly_comp,S):
    
    tensors = []
    shift = 0
    for p in poly_comp:
        
        rp = math.comb(r+p-1,p)
        idces = compute_indices([],0,r,r,p-1)
        
        T = np.zeros((r,r**p))
        T[:,idces] = S[:,shift:shift+rp]
        reshape_list = [r for _ in range (p+1)]
        tensors.append(T.reshape(*reshape_list))
        
        shift += rp
    
    return tuple(tensors)
    
        
def solve_least_squares_problem(mpi_pool,Z,Y,W,P):
    
    """
        Solves the weighted least squares problem with L2 regularization. 
        The solution is M = Y@W@Z.T@inv(Z@W@Z.T + P)
    """
    
    u, s, _ = sp.linalg.svd(Z@W@Z.T + P)
    idces = np.argwhere(s > 1e-12).reshape(-1)
    
    return Y@W@Z.T@u[:,idces]@np.diag(1./s[idces])@(u[:,idces]).T



def operator_inference(mpi_pool,Phi,poly_comp,lambdas):
    
    n, r = Phi.shape
    
    W = assemble_W(mpi_pool)
    Y = assemble_Y(mpi_pool,Phi)
    Z = assemble_Z(mpi_pool,Phi,poly_comp)
    P = assemble_P(r,poly_comp,lambdas)
    
    S = solve_least_squares_problem(mpi_pool,Z,Y,W,P)
    
    return extract_tensors(r,poly_comp,S)
    
  
