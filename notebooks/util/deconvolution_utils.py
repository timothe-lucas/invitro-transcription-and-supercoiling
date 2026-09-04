# deconvolution_utils.py
"""
Deconvolution Utilities for Molecular Beacon Kinetic Analysis
Author: Molecular Transcription Group
"""

import pickle
import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import savgol_filter


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


import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import savgol_filter


def compute_instantaneous_rate(
    rfu_signal: np.ndarray,
    f_kernel: np.ndarray,
    alpha_slope: float,
    dt_min: float,
    lambda_reg: float = 0.5,
    window_min: float = 50.0,
    polyorder: int = 2,
    mode: str = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates cumulative RNA mass and instantaneous transcription rate

    via Tikhonov deconvolution and Savitzky-Golay filtered differentiation.

    Parameters
    ----------
    rfu_signal : np.ndarray
        Raw RFU signal for a specific well.
    f_kernel : np.ndarray
        Deconvolution impulse response kernel.
    alpha_slope : float
        Calibration slope (RFU to nM).
    dt_min : float
        Time step between cycles (in minutes).
    lambda_reg : float, optional
        Tikhonov regularization parameter (default is 0.5).
    window_min : float, optional
        Savitzky-Golay integration window size in minutes (default is 50.0).
    polyorder : int, optional
        Polynomial order for Savitzky-Golay filtering (default is 2).
    mode : str, optional
        Edge effect handling mode for Savitzky-Golay filtering (default is 'nearest').

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - rna_cumulated (nM): Cumulative mass of RNA produced.
        - rate (nM/min): Instantaneous transcription rate.
    """
    N = len(rfu_signal)

    # 1. Normalization and zero-baseline alignment of the RFU signal
    S_signal = (rfu_signal - rfu_signal[0]) / alpha_slope
    S_signal = np.maximum(S_signal, 0.0)

    # 2. Construction of the Toeplitz matrix F
    col_1 = f_kernel[:N] / alpha_slope
    row_1 = np.zeros(N)
    row_1[0] = col_1[0]
    F_well = toeplitz(col_1, row_1)

    # 3. Tikhonov deconvolution
    x_instant, _ = solve_tikhonov_deconvolution(
        S_signal, F_well, dt=dt_min, lambda_reg=lambda_reg
    )

    # 4. Cumulative integration (RNA mass in nM)
    rna_cumulated = np.cumsum(x_instant)

    # 5. Smoothing window size computation
    window_len = int(window_min / dt_min)
    if window_len % 2 == 0:
        window_len += 1

    # 6. Savitzky-Golay differentiation (Instantaneous rate in nM/min)
    rate = savgol_filter(
        rna_cumulated,
        window_length=window_len,
        polyorder=polyorder,
        deriv=1,
        delta=dt_min,
        mode=mode,
    )
    rate = np.maximum(rate, 0.0)

    return rna_cumulated, rate