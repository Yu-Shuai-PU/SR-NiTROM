# SR-NiTROM

**Non-Intrusive Reduced-Order Modeling of Shift-Equivariant Systems via Symmetry Reduction and Trajectory-Based Optimization**

---

## 📋 Prerequisites

Before running the code, ensure you have the following dependencies installed:

- **Python 3.x**
- NumPy
- SciPy
- Matplotlib
- MPI4Py
- **PyManopt** (See installation below)

## 🛠 Installation

This package has been tested on **macOS (M4 chip)**.

### Installing PyManopt with Backend
You need to install `pymanopt` along with a backend for automatic differentiation. Choose **one** of the following backends based on your preference:

* `autograd` (Recommended)
* `torch`
* `jax`
* `tensorflow`

Run the following command in your terminal:

```bash
# Syntax: pip3 install pymanopt <backend>
# Example using Autograd (Recommended):
pip3 install pymanopt autograd
```

### Verify Installation
To confirm that the installation was successful, check the installed packages:

```bash
pip3 list | grep -E "pymanopt|autograd|torch|jax|tensorflow"
```

> For further reference, please visit the [PyManopt Documentation](https://pymanopt.org).

## 🚀 Running the Demo

1.  Navigate to the examples directory:
    ```bash
    cd Examples/<Subfolder_Name>
    ```
    *(Replace `<Subfolder_Name>` with the specific example folder you wish to run)*

2.  Run the main script using MPI:
    ```bash
    mpiexec -n <N_CPU> python3 main.py
    ```

    **Note:** Replace `<N_CPU>` with the number of CPU processors you want to use.
    * *Constraint:* The number of processors must be **less than** the number of input trajectories.

3.  For testing of trained ROMs:
    We are now finishing cleanups. To be published for usage on Friday.

## 📚 Reference

This work implements the **SR-NiTROM** method. It builds upon the original NiTROM algorithm:

> **Padovan, A., Vollmer, B. and Bodony, D. J. (2024).**
> Data-driven model reduction via non-intrusive optimization of projection operators and reduced-order dynamics.
> *SIAM J. Appl. Dyn. Syst.* 23(4): 3052-3076.
> [https://doi.org/10.1137/24M1628414](https://doi.org/10.1137/24M1628414)