from scipy.signal import detrend
import numpy as np

def rolling_std(x, w):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    cumsum2 = np.cumsum(np.insert(x**2, 0, 0))

    mean = (cumsum[w:] - cumsum[:-w]) / w
    mean2 = (cumsum2[w:] - cumsum2[:-w]) / w

    var = mean2 - mean**2
    return np.sqrt(var)

def segment_stable_variance(S, timeline, dt,
                           window=5,
                           thresh=0.05,
                           gap_factor=1.5,
                           min_length=4):

    nc, nt = S.shape

    t = (timeline - timeline[0]) / np.timedelta64(1, 'm')

    # --- gaps ---
    dt_measured = np.diff(t)
    gap_mask = dt_measured > dt * gap_factor
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

def segment_stable_phases(
    S, timeline, dt,
    window=7,
    std_thresh_rel=0.05,
    slope_thresh=0.002,
    amp_thresh=0.08,
    gap_factor=1.5,
    min_length=5,
    detrend_signal=True
):
    """
    Robust detection of quasi-stationary shading phases.

    Criteria:
    - low local variability (rolling std)
    - low slope (no drift)
    - bounded amplitude within segment
    """
    from scipy.signal import detrend

    nc, nt = S.shape
    t = (timeline - timeline[0]) / np.timedelta64(1, 'm')

    # --- detect gaps ---
    dt_measured = np.diff(t)
    gap_mask = dt_measured > dt * gap_factor
    split_idx = np.where(gap_mask)[0] + 1
    blocks = np.split(np.arange(nt), split_idx)

    durations_all, meanS_all = [], []
    plots_all, starts_all, ends_all = [], [], []

    for i in range(nc):

        S_i = S[i]

        for block in blocks:

            if len(block) < window:
                continue

            S_block = S_i[block]
            t_block = t[block]

            # --- detrend (remove slow drift) ---
            if detrend_signal:
                S_proc = detrend(S_block)
            else:
                S_proc = S_block.copy()

            # --- rolling std ---
            std = rolling_std(S_proc, window)

            pad = window // 2
            std_full = np.full(len(S_block), np.nan)
            std_full[pad:-pad] = std

            # --- relative std (scale independent) ---
            std_rel = std_full / (np.abs(S_block) + 1e-6)

            # --- slope ---
            slope = np.abs(np.gradient(S_block))

            # --- combined stability criterion ---
            stable = (
                (std_rel < std_thresh_rel) &
                (slope < slope_thresh)
            )
            stable[np.isnan(stable)] = False

            # --- segmentation ---
            changes = np.diff(stable.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            if stable[0]:
                starts = np.insert(starts, 0, 0)
            if stable[-1]:
                ends = np.append(ends, len(stable))

            for s, e in zip(starts, ends):

                if (e - s) < min_length:
                    continue

                segment = S_block[s:e]

                # --- amplitude filter (key improvement) ---
                if np.max(segment) - np.min(segment) > amp_thresh:
                    continue

                durations_all.append(t_block[e - 1] - t_block[s])
                meanS_all.append(segment.mean())
                plots_all.append(i)
                starts_all.append(block[s])
                ends_all.append(block[e - 1] + 1)

    return (
        np.array(durations_all),
        np.array(meanS_all),
        np.array(plots_all),
        np.array(starts_all),
        np.array(ends_all)
    )


def split_into_blocks(timeline, dt, gap_factor=1.5):
    """
    Split timeline into contiguous blocks (e.g. days)
    """
    t = (timeline - timeline[0]) / np.timedelta64(1, 'm')

    dt_measured = np.diff(t)
    gap_mask = dt_measured > dt * gap_factor

    split_idx = np.where(gap_mask)[0] + 1
    blocks = np.split(np.arange(len(timeline)), split_idx)

    return blocks



def compute_features_blockwise(S, timeline, dt, window=7, detrend_signal=True):

    nc, nt = S.shape
    blocks = split_into_blocks(timeline, dt)

    std_rel_list = [np.full(nt, np.nan) for _ in range(nc)]
    slope_list = [np.full(nt, np.nan) for _ in range(nc)]

    def rolling_std(x, w):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        cumsum2 = np.cumsum(np.insert(x**2, 0, 0))
        mean = (cumsum[w:] - cumsum[:-w]) / w
        mean2 = (cumsum2[w:] - cumsum2[:-w]) / w
        return np.sqrt(mean2 - mean**2)

    for i in range(nc):

        S_i = S[i]

        for block in blocks:

            if len(block) < window:
                continue

            S_block = S_i[block]

            # detrend local (important)
            if detrend_signal:
                S_proc = detrend(S_block)
            else:
                S_proc = S_block.copy()

            std = rolling_std(S_proc, window)

            pad = window // 2
            std_full = np.full(len(block), np.nan)
            std_full[pad:-pad] = std

            std_rel = std_full / (np.abs(S_block) + 1e-6)
            slope = np.abs(np.gradient(S_block))

            std_rel_list[i][block] = std_rel
            slope_list[i][block] = slope

    std_rel_all = np.concatenate(std_rel_list)
    slope_all = np.concatenate(slope_list)

    return std_rel_all, slope_all, std_rel_list, slope_list, blocks

from sklearn.cluster import KMeans

def calibrate_kmeans(std_rel_all, slope_all):

    mask = (~np.isnan(std_rel_all)) & (~np.isnan(slope_all))

    X = np.vstack([std_rel_all[mask], slope_all[mask]]).T
    X_log = np.log10(X + 1e-12)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_log)

    labels = kmeans.labels_
    centers = 10**kmeans.cluster_centers_

    stable_cluster = np.argmin(np.sum(centers, axis=1))
    stable_points = X[labels == stable_cluster]

    std_thresh = np.max(stable_points[:, 0])
    slope_thresh = np.max(stable_points[:, 1])

    return std_thresh, slope_thresh, kmeans



def segment_phases_blockwise(
    S, timeline, dt,
    std_rel_list, slope_list,
    std_thresh, slope_thresh,
    amp_thresh=0.08,
    min_length=5
):

    nc, nt = S.shape
    t = (timeline - timeline[0]) / np.timedelta64(1, 'm')
    blocks = split_into_blocks(timeline, dt)

    durations_all, meanS_all = [], []
    plots_all, starts_all, ends_all = [], [], []

    for i in range(nc):

        S_i = S[i]
        std_rel = std_rel_list[i]
        slope = slope_list[i]

        for block in blocks:

            stable = (
                (std_rel[block] < std_thresh) &
                (slope[block] < slope_thresh)
            )

            stable[np.isnan(stable)] = False

            changes = np.diff(stable.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            if stable[0]:
                starts = np.insert(starts, 0, 0)
            if stable[-1]:
                ends = np.append(ends, len(stable))

            for s, e in zip(starts, ends):

                if (e - s) < min_length:
                    continue

                segment = S_i[block][s:e]

                if np.max(segment) - np.min(segment) > amp_thresh:
                    continue

                durations_all.append(t[block][e - 1] - t[block][s])
                meanS_all.append(segment.mean())
                plots_all.append(i)
                starts_all.append(block[s])
                ends_all.append(block[e - 1] + 1)

    return (
        np.array(durations_all),
        np.array(meanS_all),
        np.array(plots_all),
        np.array(starts_all),
        np.array(ends_all)
    )

def estimate_slope_threshold(slope_all, q=20):
    """
    Estimate slope threshold from data (quantile-based)
    """
    slope_all = slope_all[~np.isnan(slope_all)]
    return np.nanpercentile(slope_all, q)

def grow_stable_regions(core, extend_mask):
    """
    Extend core stable regions using relaxed condition
    """

    stable = core.copy()
    n = len(core)

    i = 0
    while i < n:

        if not core[i]:
            i += 1
            continue

        # --- trouver segment noyau ---
        start = i
        while i < n and core[i]:
            i += 1
        end = i

        # --- extension à gauche ---
        j = start - 1
        while j >= 0 and extend_mask[j]:
            stable[j] = True
            j -= 1

        # --- extension à droite ---
        j = end
        while j < n and extend_mask[j]:
            stable[j] = True
            j += 1

    return stable

def segment_phases_region_growing(
    S, timeline, dt,
    std_rel_list, slope_list,
    slope_thresh,
    std_rel_thresh_factor=2,
    slope_relax_factor=5,
    amp_thresh=0.08,
    min_length=5
):

    nc, nt = S.shape
    t = (timeline - timeline[0]) / np.timedelta64(1, 'm')
    blocks = split_into_blocks(timeline, dt)

    durations_all, meanS_all = [], []
    plots_all, starts_all, ends_all = [], [], []

    slope_loose = slope_thresh * slope_relax_factor

    for i in range(nc):

        S_i = S[i]
        std_rel = std_rel_list[i]
        slope = slope_list[i]

        for block in blocks:

            std_block = std_rel[block]
            slope_block = slope[block]
            S_block = S_i[block]

            # --- noyaux stricts ---
            core = slope_block < slope_thresh

            # --- seuil std local (adaptatif) ---
            std_valid = std_block[~np.isnan(std_block)]
            if len(std_valid) == 0:
                continue

            std_thresh = np.nanpercentile(std_valid, 50) * std_rel_thresh_factor

            # --- extension ---
            extend_mask = (
                (slope_block < slope_loose) &
                (std_block < std_thresh)
            )

            stable = grow_stable_regions(core, extend_mask)
            stable[np.isnan(stable)] = False

            # --- segmentation ---
            changes = np.diff(stable.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            if stable[0]:
                starts = np.insert(starts, 0, 0)
            if stable[-1]:
                ends = np.append(ends, len(stable))

            for s, e in zip(starts, ends):

                if (e - s) < min_length:
                    continue

                segment = S_block[s:e]

                # filtre amplitude
                if np.max(segment) - np.min(segment) > amp_thresh:
                    continue

                durations_all.append(t[block][e - 1] - t[block][s])
                meanS_all.append(segment.mean())
                plots_all.append(i)
                starts_all.append(block[s])
                ends_all.append(block[e - 1] + 1)

    return (
        np.array(durations_all),
        np.array(meanS_all),
        np.array(plots_all),
        np.array(starts_all),
        np.array(ends_all)
    )

def run_stable_phase_detection(
    S, timeline, dt,
    window=7,
    amp_thresh=0.08,
    min_length=5,
    detrend_signal=True,
    plot=True
):
    """
    Full pipeline:
    - block-aware feature computation
    - KMeans calibration
    - segmentation
    - optional visualization
    """

    # --- 1. features ---
    std_rel_all, slope_all, std_rel_list, slope_list, blocks = \
        compute_features_blockwise(
            S, timeline, dt,
            window=window,
            detrend_signal=detrend_signal
        )

    # --- 2. calibration ---
    std_thresh, slope_thresh, kmeans = \
        calibrate_kmeans(std_rel_all, slope_all)

    # --- 3. segmentation ---
    results = segment_phases_blockwise(
        S, timeline, dt,
        std_rel_list, slope_list,
        std_thresh, slope_thresh,
        amp_thresh=amp_thresh,
        min_length=min_length
    )


    return {
        "durations": results[0],
        "means": results[1],
        "plots": results[2],
        "starts": results[3],
        "ends": results[4],
        "std_thresh": std_thresh,
        "slope_thresh": slope_thresh,
        "kmeans": kmeans,
        "blocks": blocks,
        "std_rel_all": std_rel_all,
        "slope_all": slope_all
    }    


def run_stable_phase_detection(
    S, timeline, dt,
    window=7,
    slope_quantile=20,
    plot=False
):

    # --- features ---
    std_rel_all, slope_all, std_rel_list, slope_list, blocks = \
        compute_features_blockwise(S, timeline, dt, window=window)

    # --- seuil slope ---
    slope_thresh = estimate_slope_threshold(
        slope_all,
        q=slope_quantile
    )

    # --- segmentation ---
    results = segment_phases_region_growing(
        S, timeline, dt,
        std_rel_list, slope_list,
        slope_thresh
    )

    return {
        "durations": results[0],
        "means": results[1],
        "plots": results[2],
        "starts": results[3],
        "ends": results[4],
        "slope_thresh": slope_thresh
    }

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

def plot_stable_phase_histogram(stable_phases, dt):
    values, counts = np.unique(stable_phases["durations"], return_counts=True)
    values, counts = zip(*sorted(zip(values, counts), key=lambda v:v[0]))
    print(dt*0.8)
    plt.bar(values, counts, (dt/60)*0.8)
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