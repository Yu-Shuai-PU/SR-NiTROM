# SR-NiTROM

**Non-Intrusive Reduced-Order Modeling of Shift-Equivariant Systems via Symmetry Reduction and Trajectory-Based Optimization**

---

## 📋 Prerequisites

Before running the code, ensure you have the following dependencies installed:

- **Python 3.x**
- NumPy
- SciPy
- Matplotlib
- MPI4Py (https://mpi4py.readthedocs.io, along with OpenMPI https://www.open-mpi.org)
- PyManopt (https://pymanopt.org)

## 🛠 Installation

This package has been tested on **macOS (M4 chip)**. To install the required libraries, please type the following commands in the terminal:

```bash
pip3 install numpy scipy matplotlib mpi4py openmpi
pip3 install pymanopt autograd
```

Here, the argument `autograd` is one of the backends of `pymanopt` for automatic differentiation. Recommended choices are shown below:

* `autograd` (Recommended)
* `torch`
* `jax`
* `tensorflow`

To confirm that the installation of PyManopt was successful, check the installed packages:

```bash
pip3 list | grep -E "pymanopt|autograd|torch|jax|tensorflow"
```

## 🚀 Running the Demo

1.  Navigate to the examples directory:
    ```bash
    cd Examples/<Subfolder_Name>
    ```
    *(Replace `<Subfolder_Name>` with the specific example folder you wish to run, **up to now only the "kse" subfolder is available for demo**)*

2.  Run the scripts using MPI to generate training data and train the model:
    ```bash
    mpiexec -n <N_CPU> python3 generate_training_data.py
    mpiexec -n <N_CPU> python3 train_SR_NiTROM.py
    mpiexec -n <N_CPU> python3 generate_testing_data.py
    mpiexec -n <N_CPU> python3 test_SR_NiTROM.py
    python3 plot.py
    ```
    During the process, you will then see several automatically created folders containing trained coefficients, output trajectories and figures for training and testing dataset.

    **Note:** Replace `<N_CPU>` with the number of CPU processors you want to use.
    * *Constraint:* The number of processors must be **less than** the number of input trajectories.

3. Tips of the training for SR-NiTROM:

We are using curriculum learning method to solve the non-convex optimization problem of SR-NiTROM. To this end, we first extract POD-Galerkin bases and
coefficients as the initial guess over the entire timespan of training trajectories. Afterwards, we train iteratively starting from short trajectory slices to longer ones 
to help maintain a robust training process. 

To adjust the timespan of trajectory for training SR-NiTROM coefficients from its initial guess, navigate to /Examples/kse/train_SR_NiTROM.py and modify these variables:
    ```python
    timespan_percentage_POD = 1.00 # percentage of the entire timespan used for POD
    timespan_percentage_NiTROM_training = 0.025 # percentage of the entire timespan used for NiTROM training
    ```

To reproduce results, please use our benchmark 10-dimensional SR-NiTROM for model reduction. Its bases and coefficients are stored in /Examples/kse/archive_benchmark_10_modes for that specific system. 

## 📚 Reference

This work implements the **SR-NiTROM** method. It builds upon the original NiTROM algorithm:

> **Padovan, A., Vollmer, B. and Bodony, D. J. (2024).**
> Data-driven model reduction via non-intrusive optimization of projection operators and reduced-order dynamics.
> *SIAM J. Appl. Dyn. Syst.* 23(4): 3052-3076.
> [https://doi.org/10.1137/24M1628414](https://doi.org/10.1137/24M1628414)