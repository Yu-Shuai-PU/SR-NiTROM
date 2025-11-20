# SR-NiTROM
Non-Intrusive Reduced-Order Modeling of Shift-Equivariant Systems via Symmetry Reduction and Trajectory-Based Optimization

# Prerequisities
NumPy
SciPy
Matplotlib
MPI4Py
PyManopt

# Installation of PyManopt (tested on M4 MacOS)
In terminal, type:
    "pip3 install pymanopt ###"
where ### = ['autograd','torch','jax','tensorflow'].

To verify the installation, type in terminal:
    "pip3 list | grep -E "pymanopt|###"".

For further reference, see the documentation of PyManopt https://pymanopt.org

# Running
To run a demo, navigate to /Examples/... where you shall see several examples.
Set one of the subfolder as the working directory.
From there, open "main.py", and type in terminal:
    "mpiexec -n ## python3 main.py",
where ## = the number of CPU processors to be used.
It is required that the number of processors should be less than the number of trajectories to be input.

# Related paper
Here is the paper of the original Non-intrusive Reduced-Order Modeling via Trajectory-based Optimization (NiTROM) algorithm:

Padovan, A., Vollmer, B. and Bodony, D. J. (2024). Data-driven model reduction via non-intrusive optimization of projection operators and reduced-order dynamics. SIAM J. Appl. Dyn. Syst. 23(4): 3052-3076. https://doi.org/10.1137/24M1628414
