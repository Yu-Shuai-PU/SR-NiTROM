import numpy as np
import matplotlib.pyplot as plt

# when n = 8 for KSE model reduction of training trajectories

x = np.arange(9)

relative_error_SRG = [1.2053, 1.2181, 1.3263, 1.2880, 1.1768, np.inf, 1.2016, 1.3730, 1.2583]

relative_error_SRG_fitted = [0.6911, 0.7133, 0.7650, 0.7601, 0.7494, np.inf, 0.7337, 0.8028, 0.7472]

relative_error_SRN = [0.6811, 0.4408, 0.6775, 0.5718, 0.5797, 1.1623, 0.3378, 0.5290, 0.2822]

relative_error_SRN_fitted = [0.2840, 0.4114, 0.5826, 0.4948, 0.5215, 0.7531, 0.2541, 0.3527, 0.2246]

plt.plot(x, relative_error_SRG, marker='o', label='SRG', color='blue')
plt.plot(x, relative_error_SRG_fitted, marker='o', label='SRG (fitted)', color='cyan')
plt.plot(x, relative_error_SRN, marker='o', label='SRN', color='red')
plt.plot(x, relative_error_SRN_fitted, marker='o', label='SRN (fitted)', color='orange')
plt.yscale('log')
plt.xlabel('Test Case Index')
# plt.ylabel('Relative Error (log scale)')
plt.title('Relative Error Comparison of SR-Galerkin and SR-NiTROM Methods')
plt.legend()
plt.grid(True)
plt.show()

# when n = 8 for KSE model reduction of testing trajectories

x = np.arange(4)

relative_error_SRG = [1.3381, 1.2418, 1.3138, 1.1279]

relative_error_SRG_fitted = [0.7609, 0.7912, 0.7450, 0.7161]

relative_error_SRN = [0.4382, 0.6406, 0.5633, 1.3540]

relative_error_SRN_fitted = [0.4155, 0.5569, 0.4367, 0.8196]

plt.plot(x, relative_error_SRG, marker='o', label='SRG', color='blue')
plt.plot(x, relative_error_SRG_fitted, marker='o', label='SRG (fitted)', color='cyan')
plt.plot(x, relative_error_SRN, marker='o', label='SRN', color='red')
plt.plot(x, relative_error_SRN_fitted, marker='o', label='SRN (fitted)', color='orange')
plt.yscale('log')
plt.xlabel('Test Case Index')
# plt.ylabel('Relative Error (log scale)')
plt.title('Relative Error Comparison of SR-Galerkin and SR-NiTROM Methods')
plt.legend()
plt.grid(True)
plt.show()



# when n = 16 for KSE model reduction of training trajectories

x = np.arange(9)

relative_error_SRG = [0.1751, 0.1902, 0.3172, 0.5184, 0.4448, np.inf, 0.3270, 0.1998, 0.3100]

relative_error_SRG_fitted = [0.1697, 0.1033, 0.2962, 0.3707, 0.1361, np.inf, 0.1231, 0.1349, 0.1426]

relative_error_SRN = [0.1739, 0.1183, 0.2490, 0.3717, 0.1917, 0.2578, 0.1647, 0.2082, 0.1235]

relative_error_SRN_fitted = [0.1403, 0.1025, 0.1726, 0.2970, 0.1060, 0.2550, 0.1628, 0.1054, 0.1249]

plt.plot(x, relative_error_SRG, marker='o', label='SRG', color='blue')
plt.plot(x, relative_error_SRG_fitted, marker='o', label='SRG (fitted)', color='cyan')
plt.plot(x, relative_error_SRN, marker='o', label='SRN', color='red')
plt.plot(x, relative_error_SRN_fitted, marker='o', label='SRN (fitted)', color='orange')
plt.yscale('log')
plt.xlabel('Test Case Index')
# plt.ylabel('Relative Error (log scale)')
plt.title('Relative Error Comparison of SR-Galerkin and SR-NiTROM Methods')
plt.legend()
plt.grid(True)
plt.show()