import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter
from scipy.signal import savgol_filter


def _split_into_blocks(timeline, dt, gap_factor=1.5):
    """
    Split timeline into contiguous blocks using time gaps.
    """
    t = (timeline - timeline[0]) / np.timedelta64(1, "m")
    dt_measured = np.diff(t)
    split_idx = np.where(dt_measured > dt * gap_factor)[0] + 1
    return np.split(np.arange(len(timeline)), split_idx)


def _normalize_date(date_like):
    """
    Convert input date to numpy datetime64[D].
    Accepts string like '2023-05-21' or np.datetime64.
    """
    return np.datetime64(date_like, "D")


def plot_smoothing_for_date(
    S,
    timeline,
    dt,
    date,
    plot_idx=0,
    gap_factor=1.5,
    uniform_sizes=(3, 5, 9),
    gaussian_sigmas=(1, 2),
    median_sizes=(3, 5),
    savgol_windows=(5, 9),
    savgol_polyorder=2,
    figsize=(12, 10),
    show_legend=True
):
    """
    Plot several smoothing options for one plot and one date/block.

    Parameters
    ----------
    S : ndarray of shape (n_plots, n_times)
        Signal matrix.
    timeline : ndarray of dtype datetime64
        Time vector.
    dt : float
        Expected time step in minutes.
    date : str or np.datetime64
        Date to display, e.g. '2023-05-21'.
    plot_idx : int
        Plot index in S.
    gap_factor : float
        Gap factor used to split contiguous daily blocks.
    uniform_sizes : tuple of int
        Window sizes for uniform moving average.
    gaussian_sigmas : tuple of float
        Sigmas for gaussian smoothing.
    median_sizes : tuple of int
        Window sizes for median filter.
    savgol_windows : tuple of int
        Window sizes for Savitzky-Golay filter.
    savgol_polyorder : int
        Polynomial order for Savitzky-Golay.
    figsize : tuple
        Figure size.
    show_legend : bool
        Whether to display legends.

    Returns
    -------
    fig, axes, results
        results is a dict containing the extracted block and smoothed signals.
    """

    if S.ndim != 2:
        raise ValueError("S must be a 2D array of shape (n_plots, n_times).")

    if plot_idx < 0 or plot_idx >= S.shape[0]:
        raise IndexError("plot_idx is out of bounds.")

    import pandas as pd
    date = pd.Timestamp(date).normalize()   
    blocks = _split_into_blocks(timeline, dt, gap_factor=gap_factor)

    # find blocks belonging to the requested date
    candidate_blocks = []
    for block in blocks:
        if len(block) == 0:
            continue
        block_date = timeline[block[0]].normalize()
        if block_date == date:
            candidate_blocks.append(block)

    if len(candidate_blocks) == 0:
        raise ValueError(f"No block found for date {date}.")

    # If multiple blocks in same date, concatenate for display
    idx = np.concatenate(candidate_blocks)

    x = S[plot_idx, idx]
    t = timeline[idx]

    results = {
        "timeline": t,
        "raw": x,
        "uniform": {},
        "gaussian": {},
        "median": {},
        "savgol": {}
    }

    # compute smoothings
    for w in uniform_sizes:
        if w >= 1:
            results["uniform"][w] = uniform_filter1d(x, size=w, mode="nearest")

    for sigma in gaussian_sigmas:
        if sigma > 0:
            results["gaussian"][sigma] = gaussian_filter1d(x, sigma=sigma, mode="nearest")

    for w in median_sizes:
        if w >= 1:
            if w % 2 == 0:
                w += 1
            results["median"][w] = median_filter(x, size=w, mode="nearest")

    for w in savgol_windows:
        if w < 3:
            continue
        if w % 2 == 0:
            w += 1
        if w > len(x):
            continue
        poly = min(savgol_polyorder, w - 1)
        results["savgol"][w] = savgol_filter(x, window_length=w, polyorder=poly, mode="interp")

    # plotting
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)

    axes[0].plot(t, x, label="Raw")
    axes[0].set_title(f"Plot {plot_idx} — {str(date)}")
    axes[0].set_ylabel("Shading")
    axes[0].grid(True, alpha=0.3)
    if show_legend:
        axes[0].legend()

    axes[1].plot(t, x, color="0.75", label="Raw")
    for w, y in results["uniform"].items():
        axes[1].plot(t, y, label=f"uniform w={w}")
    axes[1].set_title("Uniform smoothing")
    axes[1].set_ylabel("Shading")
    axes[1].grid(True, alpha=0.3)
    if show_legend:
        axes[1].legend()

    axes[2].plot(t, x, color="0.75", label="Raw")
    for sigma, y in results["gaussian"].items():
        axes[2].plot(t, y, label=f"gaussian σ={sigma}")
    axes[2].set_title("Gaussian smoothing")
    axes[2].set_ylabel("Shading")
    axes[2].grid(True, alpha=0.3)
    if show_legend:
        axes[2].legend()

    axes[3].plot(t, x, color="0.75", label="Raw")
    for w, y in results["median"].items():
        axes[3].plot(t, y, label=f"median w={w}")
    axes[3].set_title("Median smoothing")
    axes[3].set_ylabel("Shading")
    axes[3].grid(True, alpha=0.3)
    if show_legend:
        axes[3].legend()

    axes[4].plot(t, x, color="0.75", label="Raw")
    for w, y in results["savgol"].items():
        axes[4].plot(t, y, label=f"savgol w={w}")
    axes[4].set_title("Savitzky-Golay smoothing")
    axes[4].set_ylabel("Shading")
    axes[4].grid(True, alpha=0.3)
    if show_legend:
        axes[4].legend()

    plt.tight_layout()

    return fig, axes, results