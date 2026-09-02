# deconvolution_utils.py
"""
Deconvolution Utilities for Molecular Beacon Kinetic Analysis
Author: Molecular Transcription Group
"""

import pickle
import numpy as np
from scipy.linalg import toeplitz


def build_toeplitz_matrix(f_kernel):
    """Constructs a causal lower-triangular Toeplitz system matrix from a response kernel.

    Parameters:
        f_kernel (array-like): Discrete impulse response vector f(t).

    Returns:
        ndarray: Square lower-triangular Toeplitz matrix (N x N).
    """
    col_1 = f_kernel
    row_1 = np.zeros(len(f_kernel))
    row_1[0] = f_kernel[0]
    return toeplitz(col_1, row_1)


def solve_tikhonov_deconvolution(S_signal, F_mat, dt, lambda_reg=0.5):
    """Solves the ill-posed linear inverse problem F * x = S using L2 Tikhonov Regularization:

        x_instant = (F^T * F + lambda * I)^(-1) * F^T * S

    Parameters:
        S_signal (array-like): Observed signal vector (nM).
        F_mat (ndarray): Normalized system matrix.
        dt (float): Time step duration (minutes).
        lambda_reg (float): Tikhonov regularization parameter.

    Returns:
        tuple: (x_instant, S_deconv)
            - x_instant: Instantaneous RNA accumulation per time step (nM/step).
            - S_deconv: Reconstructed cumulative RNA concentration (nM).
    """
    Ft = F_mat.T
    I_mat = np.eye(F_mat.shape[1])

    # Regularized inverse matrix calculation
    inv_matrix = np.linalg.inv(Ft @ F_mat + lambda_reg * I_mat)

    # Calculate step-by-step production
    x_instant = inv_matrix @ Ft @ S_signal

    # Integrate step production to recover cumulative concentration
    S_deconv = np.cumsum(x_instant)

    return x_instant, S_deconv


def save_deconvolution_assets(filepath, F_matrix, f_kernel, alpha_slope, dt):
    # Convertit automatiquement un objet Path en str si besoin
    filepath_str = str(filepath)
    assets = {
        'F_matrix': F_matrix,
        'f_kernel': f_kernel,
        'alpha_slope': alpha_slope,
        'dt': dt,
    }
    with open(filepath_str, 'wb') as f:
        pickle.dump(assets, f)
    print(f"Successfully exported assets to: {filepath_str}")


def load_deconvolution_assets(filepath):
    filepath_str = str(filepath)
    with open(filepath_str, 'rb') as f:
        assets = pickle.load(f)
    print(f"Successfully loaded assets from: {filepath_str}")
    return (
        assets['F_matrix'],
        assets['f_kernel'],
        assets['alpha_slope'],
        assets['dt'],
    )