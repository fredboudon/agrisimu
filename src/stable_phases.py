from scipy.signal import detrend
import numpy as np
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter

def rolling_std_centered_edges(x, w, min_count=3):
    """
    Centered rolling std with truncated windows at edges.
    """
    n = len(x)
    half = w // 2
    out = np.full(n, np.nan, dtype=float)

    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        if (b - a) >= min_count:
            out[i] = np.std(x[a:b])

    return out


def close_small_gaps(mask, max_gap=1):
    """
    Fill short False gaps inside True regions.
    """
    mask = mask.copy()
    n = len(mask)
    i = 0

    while i < n:
        if mask[i]:
            i += 1
            continue

        j = i
        while j < n and not mask[j]:
            j += 1

        if i > 0 and j < n and (j - i) <= max_gap and mask[i - 1] and mask[j]:
            mask[i:j] = True

        i = j

    return mask


def extract_segments(mask):
    """
    Extract contiguous True segments from boolean mask.
    """
    changes = np.diff(mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if len(mask) and mask[0]:
        starts = np.insert(starts, 0, 0)
    if len(mask) and mask[-1]:
        ends = np.append(ends, len(mask))

    return starts, ends


def merge_close_segments(starts, ends, values, max_gap_pts=2, max_mean_diff=0.02):
    """
    Merge neighboring segments if:
    - gap is small
    - mean value is similar
    """
    if len(starts) == 0:
        return starts, ends

    new_starts = [starts[0]]
    new_ends = [ends[0]]

    for s, e in zip(starts[1:], ends[1:]):
        prev_s = new_starts[-1]
        prev_e = new_ends[-1]

        gap = s - prev_e
        mean_prev = values[prev_s:prev_e].mean()
        mean_curr = values[s:e].mean()

        if gap <= max_gap_pts and abs(mean_curr - mean_prev) <= max_mean_diff:
            new_ends[-1] = e
        else:
            new_starts.append(s)
            new_ends.append(e)

    return np.array(new_starts), np.array(new_ends)


def segment_stable_phases(
    S, timeline, dt,
    window=5,
    thresh=0.04,
    sigma=1.0,
    min_count=3,
    min_length=3,
    close_gap_pts=1,
    merge_gap_pts=1,
    merge_mean_diff=0.02
):
    """
    Detect stable phases from rolling std computed on a Gaussian-smoothed signal.

    Parameters
    ----------
    S : ndarray (n_plots, n_times)
    timeline : datetime-like array
    dt : float
        Expected timestep in minutes
    window : int
        Rolling std window
    thresh : float
        Threshold on rolling std
    sigma : float
        Gaussian smoothing sigma
    min_count : int
        Minimum number of points for std near block edges
    gap_factor : float
        Split blocks when measured dt > dt * gap_factor
    min_length : int
        Minimum stable segment length in number of points
    close_gap_pts : int
        Fill short unstable gaps inside stable regions
    merge_gap_pts : int
        Merge neighboring stable segments if close enough
    merge_mean_diff : float
        Maximum mean difference to merge neighboring segments

    Returns
    -------
    durations, means, plots, starts, ends
    """

    nc, nt = S.shape

    t0 = timeline[0]
    t = np.asarray((timeline - t0) / np.timedelta64(1, "m"), dtype=float)

    # --- split into contiguous blocks ---
    gap_mask = np.diff(timeline.day) != 0
    split_idx = np.where(gap_mask)[0] + 1
    assert len(split_idx) == len(np.unique(timeline.date)) - 1, "Too many gaps detected, check timeline and dt"
    blocks = np.split(np.arange(nt), split_idx)

    durations_all = []
    meanS_all = []
    plots_all = []
    starts_all = []
    ends_all = []

    for i in range(nc):
        S_i = S[i]

        for block in blocks:
            #print(f"Block from t={t[block[0]]:.1f} to t={t[block[-1]]:.1f} min, length={len(block)} points, duration={t[block[-1]] - t[block[0]]:.1f} min")
            assert (t[block[-1]] - t[block[0]]) < 24 * 60, "Block duration seems too long, check timeline and dt"

            if len(block) < min_length:
                continue

            S_block = S_i[block]
            t_block = t[block]

            # --- smoothing before rolling std ---
            if sigma is not None and sigma > 0:
                S_smooth = gaussian_filter1d(S_block, sigma=sigma, mode="nearest")
            else:
                S_smooth = S_block.copy()

            # --- rolling std on smoothed signal ---
            std_full = rolling_std_centered_edges(S_smooth, window, min_count=min_count)

            # --- stable mask ---
            stable = std_full < thresh
            stable[np.isnan(stable)] = False

            assert len(stable) == len(S_block), "Stable mask length mismatch with block length"

            # --- fill tiny holes ---
            if close_gap_pts > 0:
                stable = close_small_gaps(stable, max_gap=close_gap_pts)

            assert len(stable) == len(S_block), "Stable mask length mismatch with block length"

            # --- segment extraction ---
            starts, ends = extract_segments(stable)


            if len(starts) == 0:
                continue

            if np.max(t_block[ends-1] - t_block[starts]) > 24 * 60:
                for s, e in zip(starts, ends):
                    print(f"  Segment from t={t_block[s]:.1f} to t={t_block[e-1]:.1f} min, duration={t_block[e-1] - t_block[s]:.1f} min in block from t={t_block[0]:.1f} to t={t_block[-1]:.1f} min")
                print(f"  Found {len(starts)} initial segments in block, max duration={np.max(t_block[ends-1] - t_block[starts]):.1f} min")
            
            assert np.max(t_block[ends-1] - t_block[starts]) < 24 * 60, "Block duration seems too long, check timeline and dt"

            # --- merge similar neighboring segments on raw signal ---
            starts, ends = merge_close_segments(
                starts, ends, S_block,
                max_gap_pts=merge_gap_pts,
                max_mean_diff=merge_mean_diff
            )
            assert np.max(t_block[ends-1] - t_block[starts]) < 24 * 60, "Block duration seems too long, check timeline and dt"

            lengths = ends - starts
            keep = lengths >= min_length
            starts = starts[keep]
            ends = ends[keep]

            if len(starts) == 0:
                continue

            durations = t_block[ends - 1] - t_block[starts]
            assert np.max(durations) < 24 * 60, "Block duration seems too long, check timeline and dt"

            means = np.array([S_block[s:e].mean() for s, e in zip(starts, ends)])

            starts_global = block[starts]
            ends_global = block[ends - 1] + 1

            durations_all.append(durations)
            meanS_all.append(means)
            plots_all.append(np.full(len(durations), i, dtype=np.int32))
            starts_all.append(starts_global)
            ends_all.append(ends_global)

    if len(durations_all) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    return (
        np.concatenate(durations_all),
        np.concatenate(meanS_all),
        np.concatenate(plots_all),
        np.concatenate(starts_all),
        np.concatenate(ends_all),
    )

def segment_stable_phases_simple(S, timeline, dt,
                           window=5,
                           thresh=0.05,
                           min_length=3):

    nc, nt = S.shape

    t = (timeline - timeline[0]) / np.timedelta64(1, 'm')

    # --- gaps ---
    gap_mask = np.diff(timeline.day) != 0
    split_idx = np.where(gap_mask)[0] + 1
    blocks = np.split(np.arange(nt), split_idx)

    durations_all = []
    meanS_all = []
    plots_all = []
    starts_all = []
    ends_all = []

    for i in range(nc):

        S_i = S[i]

        for block in blocks:

            if len(block) < window:
                continue

            S_block = S_i[block]
            t_block = t[block]

            # --- variance glissante ---
            std = rolling_std(S_block, window)

            # alignement (centré)
            pad = window // 2
            std_full = np.full(len(S_block), np.nan)
            std_full[pad:-pad] = std

            # --- seuil adaptatif ---
            #thresh = 0.05 #np.nanpercentile(std_full, q)

            stable = std_full < thresh
            stable[np.isnan(stable)] = False

            # --- segmentation ---
            changes = np.diff(stable.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            if stable[0]:
                starts = np.insert(starts, 0, 0)
            if stable[-1]:
                ends = np.append(ends, len(stable))

            lengths = ends - starts
            mask = lengths >= min_length

            starts = starts[mask]
            ends = ends[mask]

            if len(starts) == 0:
                continue

            durations = t_block[ends - 1] - t_block[starts]
            means = np.array([S_block[s:e].mean() for s, e in zip(starts, ends)])

            starts_global = block[starts]
            ends_global = block[ends - 1] + 1

            durations_all.append(durations)
            meanS_all.append(means)
            plots_all.append(np.full(len(durations), i, dtype=np.int32))
            starts_all.append(starts_global)
            ends_all.append(ends_global)

    return (
        np.concatenate(durations_all),
        np.concatenate(meanS_all),
        np.concatenate(plots_all),
        np.concatenate(starts_all),
        np.concatenate(ends_all)
    )

import numpy as np
import pandas as pd


def compute_phase_indices_per_plot(durations, plots, mean_S, timeline, n_plots=None):
    durations = np.asarray(durations, dtype=float)
    plots = np.asarray(plots)
    mean_S = np.asarray(mean_S, dtype=float)

    if durations.shape[0] != plots.shape[0]:
        raise ValueError("durations and plots must have the same length.")

    if len(timeline) < 2:
        raise ValueError("timeline must contain at least 2 timestamps.")

    timeline = pd.to_datetime(timeline)
    dt = np.diff(timeline.values).astype("timedelta64[m]").astype(float)
    T_total = np.nansum(dt)

    if n_plots is None:
        if plots.size == 0:
            raise ValueError("n_plots must be provided when plots is empty.")
        n_plots = int(np.nanmax(plots)) + 1

    rows = []

    for p in range(n_plots):
        mask = plots == p
        d = durations[mask]

        N_phases = len(d)
        T_stable = float(np.nansum(d)) if N_phases > 0 else 0.0
        stable_fraction = T_stable / T_total if T_total > 0 else np.nan
        fragmentation = (N_phases / T_stable) if T_stable > 0 else np.nan
        mean_phase_duration = (T_stable / N_phases) if N_phases > 0 else np.nan
        mean_phase_shading = np.nanmean(mean_S[mask]) if N_phases > 0 else np.nan

        rows.append({
            "plot": p,
            "N_phases": N_phases,
            "T_stable": T_stable,
            "T_total": T_total,
            "stable_fraction": stable_fraction,
            "fragmentation": fragmentation,
            "mean_phase_duration": mean_phase_duration,
            "mean_phase_shading": mean_phase_shading,
        })

    return pd.DataFrame(rows)


def build_spatial_map(df_indices, coords, column, shape=None, fill_value=np.nan):
    """
    Build a 2D spatial map for a single column.

    Parameters
    ----------
    df_indices : pandas.DataFrame
        Must contain 'plot' and the chosen column
    coords : array-like (n_plots, 2)
        coords[p] = (x, y)
    column : str
        Column name to map
    shape : tuple or None
        (ny, nx). If None, inferred from coords
    fill_value : float
        Value for missing plots

    Returns
    -------
    grid : 2D array
    """

    coords = np.asarray(coords)

    # --- infer shape ---
    if shape is None:
        nx = int(np.max(coords[:, 0])) + 1
        ny = int(np.max(coords[:, 1])) + 1
    else:
        ny, nx = shape

    grid = np.full((ny, nx), fill_value)

    for _, row in df_indices.iterrows():
        p = int(row["plot"])

        if p >= len(coords):
            continue

        x, y = coords[p]
        grid[int(y), int(x)] = row[column]

    return grid

import matplotlib.pyplot as plt

def plot_stable_phase_distribution(stable_phases):
    
    plt.scatter(stable_phases["durations"], stable_phases["means"], marker='.')
    plt.ylabel("Mean shading")
    plt.xlabel("Duration (m)")

def plot_stable_phase_distribution_grouped(stable_phases):

    durations = stable_phases["durations"]
    mean_S = stable_phases["means"]

    # --- discrétisation ---
    mean_S_rounded = np.round(mean_S, 2)

    # --- empilement ---
    data = np.stack((durations, mean_S_rounded), axis=1)

    # --- comptage vectorisé ---
    unique, counts = np.unique(data, axis=0, return_counts=True)

    # --- plot ---
    plt.scatter(unique[:,0], unique[:,1], c=counts, marker='.', s=5)
    plt.ylabel("Mean shading")
    plt.xlabel("Duration (m)")
    plt.colorbar(label="Count")

def plot_stable_phase_distribution_grouped(stable_phases):

    durations = stable_phases["durations"]
    mean_S = stable_phases["means"]

    hb = plt.hexbin(durations, mean_S, gridsize=100, mincnt=1, norm='log', cmap='jet')
    plt.colorbar(hb, label="Count (log)")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Shading")

def plot_stable_phase_histogram(stable_phases, dt, logscale=True):
    values, counts = np.unique(stable_phases["durations"], return_counts=True)
    values, counts = zip(*sorted(zip(values, counts), key=lambda v:v[0]))
    print(dt*0.8)
    plt.bar(values, counts, (dt/60)*0.8, log=logscale)
    plt.ylabel("Number of phases")
    plt.xlabel("Duration (minutes)")
    plt.tight_layout()

def plot_stable_phase(S, stable_phases, timeline, day =  '2023/06/21', plot = 2525):
    durations, mean_S, plots, starts, ends = stable_phases["durations"], stable_phases["means"], stable_phases["plots"], stable_phases["starts"], stable_phases["ends"]
    mask = (plots == plot)     
    durations = durations[mask]
    starts = starts[mask]
    ends = ends[mask]
    mean_S = mean_S[mask]    
    plt.plot(timeline, S[plot,:], alpha=0.8)
    plt.tick_params("x", rotation=30)
    plt.hlines(mean_S, timeline[starts], timeline[ends-1], color='red')
    plt.ylabel("Shading")
    plt.xlabel("Days")
    from data_util import timeline_select_dates
    mask = timeline_select_dates(timeline,day)
    plt.xlim((mask[0],mask[-1]))


def plot_feature_space_with_kmeans(std_rel_all, slope_all, kmeans,
                                  std_thresh=None, slope_thresh=None):
    """
    Plot feature space with KMeans clusters and thresholds
    """

    mask = (~np.isnan(std_rel_all)) & (~np.isnan(slope_all))

    X = np.vstack([
        std_rel_all[mask],
        slope_all[mask]
    ]).T

    X_log = np.log10(X + 1e-12)

    labels = kmeans.predict(X_log)

    plt.figure(figsize=(6, 5))

    # scatter par cluster
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(X[idx, 0], X[idx, 1],
                    s=5, alpha=0.4, label=f"Cluster {lab}")

    # centres
    centers = 10**kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1],
                marker='x', s=100, linewidths=2, label="Centers")

    # seuils
    if std_thresh is not None:
        plt.axvline(std_thresh, linestyle='--', label="std_thresh")
    if slope_thresh is not None:
        plt.axhline(slope_thresh, linestyle='--', label="slope_thresh")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Relative std")
    plt.ylabel("Slope")
    plt.title("Feature space with KMeans clustering")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # ---------------------------------------------------------------------------
# Exemple d'utilisation / test rapide
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    nt = 500
    timeline = np.array(
        [np.datetime64("2023-06-21T07:00") + np.timedelta64(int(i * 2), "m") for i in range(nt)]
    )

    # Signal synthétique : deux phases stables + bruit + transitoire
    S_row = np.concatenate([
        np.full(80, 0.28) + rng.normal(0, 0.005, 80),   # phase stable ~0.28
        np.linspace(0.28, 0.15, 40) + rng.normal(0, 0.01, 40),  # transition
        rng.uniform(0.10, 0.42, 200),                   # phase variable
        np.full(80, 0.22) + rng.normal(0, 0.005, 80),   # phase stable ~0.22
        rng.uniform(0.05, 0.30, 100),                   # fin variable
    ])
    S = S_row[np.newaxis, :]  # shape (1, nt)

    durations, meanS, plots, starts, ends = segment_stable_phases(
        S, timeline, dt=2
    )

    t_min = (timeline - timeline[0]) / np.timedelta64(1, "m")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_min, S_row, color="steelblue", lw=0.8, label="Ombrage")
    for s, e, m in zip(starts, ends, meanS):
        ax.axhspan(m - 0.005, m + 0.005, xmin=t_min[s] / t_min[-1],
                   xmax=t_min[e - 1] / t_min[-1], alpha=0.15, color="red")
        ax.hlines(m, t_min[s], t_min[e - 1], colors="red", lw=2)
    ax.set_xlabel("Temps (min)")
    ax.set_ylabel("Ombrage")
    ax.set_title("Phases stables détectées")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/test_stable_phases.png", dpi=120)
    print(f"{len(starts)} phases stables détectées.")
    for k in range(len(starts)):
        print(f"  Segment {k+1}: t={t_min[starts[k]]:.0f}–{t_min[ends[k]-1]:.0f} min, "
              f"ombrage moyen={meanS[k]:.3f}, durée={durations[k]:.0f} min")