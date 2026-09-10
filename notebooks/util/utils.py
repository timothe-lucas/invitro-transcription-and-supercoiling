
#utils.py
"""
Deconvolution Utilities for Molecular Beacon Kinetic Analysis
Author: Timothé LUCAS
"""

import pickle

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import savgol_filter


# ==============================================================================
# DECONVOLUTION
# ==============================================================================

def solve_tikhonov_deconvolution(
    S_signal,
    F_mat,
    dt,
    lambda_reg=0.5,
):
    """Solve the ill-posed linear inverse problem F * x = S
    using L2 Tikhonov regularization.

    The regularized solution is:

        x = (F.T @ F + lambda * I)^(-1) @ F.T @ S

    Parameters
    ----------
    S_signal : array-like
        Observed signal vector (nM).
    F_mat : ndarray
        Normalized system matrix.
    dt : float
        Time step duration (minutes).
    lambda_reg : float, optional
        Tikhonov regularization parameter. Default is 0.5.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x_instant :
            Instantaneous RNA accumulation per time step (nM/step).
        S_deconv :
            Reconstructed cumulative RNA concentration (nM).
    """
    S_signal = np.asarray(S_signal, dtype=float)
    F_mat = np.asarray(F_mat, dtype=float)

    Ft = F_mat.T
    I_mat = np.eye(F_mat.shape[1])

    # Regularized inverse matrix
    inv_matrix = np.linalg.inv(
        Ft @ F_mat + lambda_reg * I_mat
    )

    # Step-by-step RNA production
    x_instant = inv_matrix @ Ft @ S_signal

    # Cumulative RNA production
    S_deconv = np.cumsum(x_instant)

    return x_instant, S_deconv


# ==============================================================================
# DECONVOLUTION ASSETS
# ==============================================================================

def save_deconvolution_assets(
    filepath,
    F_matrix,
    f_kernel,
    alpha_slope,
    dt,
):
    """Save deconvolution assets to a pickle file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Output pickle file path.
    F_matrix : ndarray
        System matrix.
    f_kernel : ndarray
        Deconvolution impulse response kernel.
    alpha_slope : float
        Calibration slope.
    dt : float
        Time step duration (minutes).
    """
    filepath_str = str(filepath)

    assets = {
        "F_matrix": F_matrix,
        "f_kernel": f_kernel,
        "alpha_slope": alpha_slope,
        "dt": dt,
    }

    with open(filepath_str, "wb") as f:
        pickle.dump(assets, f)

    print(f"Successfully exported assets to: {filepath_str}")


def load_deconvolution_assets(filepath):
    """Load deconvolution assets from a pickle file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Input pickle file path.

    Returns
    -------
    tuple
        F_matrix, f_kernel, alpha_slope, dt
    """
    filepath_str = str(filepath)

    with open(filepath_str, "rb") as f:
        assets = pickle.load(f)

    print(f"Successfully loaded assets from: {filepath_str}")

    return (
        assets["F_matrix"],
        assets["f_kernel"],
        assets["alpha_slope"],
        assets["dt"],
    )


# ==============================================================================
# INSTANTANEOUS TRANSCRIPTION RATE
# ==============================================================================

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
    """Calculate cumulative RNA mass and instantaneous transcription rate.

    The RFU signal is first deconvoluted using Tikhonov regularization.
    The cumulative RNA signal is then differentiated using a
    Savitzky-Golay filter.

    Parameters
    ----------
    rfu_signal : np.ndarray
        Raw RFU signal for a specific well.
    f_kernel : np.ndarray
        Deconvolution impulse response kernel.
    alpha_slope : float
        Calibration slope (RFU to nM).
    dt_min : float
        Time step between cycles (minutes).
    lambda_reg : float, optional
        Tikhonov regularization parameter. Default is 0.5.
    window_min : float, optional
        Savitzky-Golay window size in minutes. Default is 50.0.
    polyorder : int, optional
        Polynomial order for Savitzky-Golay filtering. Default is 2.
    mode : str, optional
        Edge handling mode for Savitzky-Golay filtering.
        Default is "nearest".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        rna_cumulated :
            Cumulative RNA mass (nM).
        rate :
            Instantaneous transcription rate (nM/min).
    """
    rfu_signal = np.asarray(rfu_signal, dtype=float)
    f_kernel = np.asarray(f_kernel, dtype=float)

    N = len(rfu_signal)

    # --------------------------------------------------------------------------
    # 1. Normalize RFU signal and align the baseline
    # --------------------------------------------------------------------------
    S_signal = (
        rfu_signal - rfu_signal[0]
    ) / alpha_slope

    S_signal = np.maximum(S_signal, 0.0)

    # --------------------------------------------------------------------------
    # 2. Construct Toeplitz system matrix
    # --------------------------------------------------------------------------
    col_1 = f_kernel[:N] / alpha_slope

    row_1 = np.zeros(N)
    row_1[0] = col_1[0]

    F_well = toeplitz(col_1, row_1)

    # --------------------------------------------------------------------------
    # 3. Tikhonov deconvolution
    # --------------------------------------------------------------------------
    x_instant, _ = solve_tikhonov_deconvolution(
        S_signal,
        F_well,
        dt=dt_min,
        lambda_reg=lambda_reg,
    )

    # --------------------------------------------------------------------------
    # 4. Cumulative RNA production
    # --------------------------------------------------------------------------
    rna_cumulated = np.cumsum(x_instant)

    # --------------------------------------------------------------------------
    # 5. Savitzky-Golay window
    # --------------------------------------------------------------------------
    window_len = int(window_min / dt_min)

    if window_len % 2 == 0:
        window_len += 1

    # Make sure the window is valid for the signal length
    if window_len > N:
        window_len = N if N % 2 == 1 else N - 1

    if window_len <= polyorder:
        raise ValueError(
            "Savitzky-Golay window is too short for the selected "
            "polynomial order."
        )

    # --------------------------------------------------------------------------
    # 6. Instantaneous transcription rate
    # --------------------------------------------------------------------------
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


# ==============================================================================
# RNAP INACTIVATION CORRECTION
# ==============================================================================

def correct_rnap_inactivation(
    rate,
    time_min,
    k_inact,
    k_inact_se=None,
):
    """Correct the instantaneous transcription rate for RNAP inactivation.

    The correction assumes a first-order RNAP inactivation process:

        v_corr(t) = v(t) * exp(k_inact * t)

    where:

        v(t)       = measured/deconvoluted transcription rate
        k_inact    = RNAP inactivation rate constant (min^-1)
        t          = time (min)
        v_corr(t)  = transcription rate corrected for RNAP inactivation

    If the standard error of k_inact is supplied, its contribution to the
    uncertainty of the corrected rate is propagated as:

        sigma_vcorr =
            v(t) * t * exp(k_inact * t) * sigma_k

    This is the same correction and uncertainty propagation used in the
    RNAP inactivation analysis notebook.

    Parameters
    ----------
    rate : array-like
        Instantaneous transcription rate v(t) (nM/min).
    time_min : array-like
        Time vector corresponding to rate (min).
    k_inact : float or array-like
        RNAP inactivation rate constant (min^-1).

        This can be:
        - a scalar, for one experiment;
        - an array with the same length as rate, if k_inact is already
          associated with every data point.

    k_inact_se : float or array-like, optional
        Standard error associated with k_inact (min^-1).

        If None, only the corrected rate is returned and the uncertainty
        is set to None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        corrected_rate :
            RNAP-inactivation-corrected transcription rate (nM/min).

        corrected_rate_se :
            Propagated uncertainty due to k_inact SE (nM/min).
            Returns None if k_inact_se is None.
            
    """
    rate = np.asarray(rate, dtype=float)
    time_min = np.asarray(time_min, dtype=float)

    if rate.shape != time_min.shape:
        raise ValueError(
            "rate and time_min must have the same shape."
        )

    if np.any(~np.isfinite(rate)):
        raise ValueError(
            "rate contains NaN or infinite values."
        )

    if np.any(~np.isfinite(time_min)):
        raise ValueError(
            "time_min contains NaN or infinite values."
        )

    # --------------------------------------------------------------------------
    # Corrected transcription rate
    # --------------------------------------------------------------------------
    correction_factor = np.exp(
        np.asarray(k_inact, dtype=float) * time_min
    )

    corrected_rate = rate * correction_factor

    # --------------------------------------------------------------------------
    # Propagate uncertainty from k_inact
    # --------------------------------------------------------------------------
    if k_inact_se is None:
        corrected_rate_se = None

    else:
        k_inact_se = np.asarray(k_inact_se, dtype=float)

        corrected_rate_se = (
            np.abs(rate)
            * np.abs(time_min)
            * correction_factor
            * np.abs(k_inact_se)
        )

    return corrected_rate, corrected_rate_se

