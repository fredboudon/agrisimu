"""Daylight duration statistics and weather visualizations."""

import pandas as pd
import numpy as np

def compute_daylength_from_timestamps(
    timestamps,
    max_gap="2H",
    infer_dt=True
):
    """
    Compute daylight duration per day from a series of timestamps.

    Parameters
    ----------
    timestamps : array-like or pandas.DatetimeIndex or pandas.Series
        Timestamps corresponding ONLY to daylight periods.
    max_gap : str or Timedelta
        Maximum gap allowed between consecutive timestamps to be considered continuous daylight.
        Larger gaps are ignored (e.g., night or missing data).
    infer_dt : bool
        If True, estimate a typical timestep to add at the end of each day.

    Returns
    -------
    pandas.Series
        Index: date
        Values: daylight duration in hours
    """

    # --- ensure datetime index ---
    ts = pd.to_datetime(timestamps)
    ts = pd.Series(index=ts, data=1).sort_index()

    if len(ts) < 2:
        raise ValueError("Need at least 2 timestamps")

    # --- compute diffs ---
    diffs = ts.index.to_series().diff()

    # remove negative or zero diffs
    diffs = diffs[diffs > pd.Timedelta(0)]

    if len(diffs) == 0:
        raise ValueError("No valid time differences found")

    # --- typical timestep ---
    dt_typical = diffs.median() if infer_dt else pd.Timedelta(0)

    max_gap = pd.to_timedelta(max_gap)

    # --- assign each diff to its day ---
    df = pd.DataFrame({
        "time": ts.index,
        "diff": diffs.reindex(ts.index)
    })

    df["date"] = df["time"].dt.date

    # --- keep only "continuous daylight" gaps ---
    df["valid"] = df["diff"] <= max_gap

    # sum valid intervals per day
    daylength = (
        df.loc[df["valid"]]
        .groupby("date")["diff"]
        .sum()
    )

    # --- add one timestep per day (optional correction) ---
    if infer_dt:
        daylength = daylength + dt_typical

    # convert to hours
    daylength_h = daylength.dt.total_seconds() / 3600.0

    return pd.Series(daylength_h, name="daylength_h")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def decimal_hours_to_hhmm(x, pos=None):
    """Convert decimal hours to HH:MM format"""
    h = int(x)
    m = int(round((x - h) * 60))
    if m == 60:  # Adjust for rounding
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}"


def violin_daylength_from_df(
    df,
    datetime_col=None,
    groupby="month",
    figsize=(5, 5),
):
    # --- timestamps ---
    if datetime_col:
        timestamps = pd.to_datetime(df[datetime_col])
    else:
        timestamps = pd.to_datetime(df.index)

    # --- compute daylength ---
    daylength = compute_daylength_from_timestamps(timestamps)

    daylength = daylength.reset_index()
    daylength.columns = ["date", "daylength_h"]
    daylength["date"] = pd.to_datetime(daylength["date"])
    daylength["month"] = daylength["date"].dt.month

    # --- grouping ---
    if groupby == "month":
        groups = [g["daylength_h"].values for _, g in daylength.groupby("month")]
        positions = sorted(daylength["month"].unique())
        labels = [pd.Timestamp(2000, m, 1).strftime("%b") for m in positions]

    elif groupby == "overall":
        groups = [daylength["daylength_h"].values]
        positions = [1]
        labels = ["All days"]

    else:
        raise ValueError("groupby must be 'month' or 'overall'")

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    ax.violinplot(groups, positions=positions, showmeans=True, showmedians=True)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Day length (HH:MM)")
    ax.set_title("Daylight duration distribution")

    # ✨ Format the y-axis as HH:MM
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(decimal_hours_to_hhmm))

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return daylength, ax

def violinplot_from_df(
    df,
    value_col,
    datetime_col=None,
    groupby="month",
    figsize=(10, 5),
    showmeans=True,
    showmedians=False,
    showextrema=True,
    ylabel=None,
    title=None,
    y_as_hhmm=False,
    ax=None,
):
    """
    Generic violin plot for a dataframe variable indexed by datetime.

    Parameters
    ----------
    df : pandas.DataFrame
    value_col : str
        Column to plot.
    datetime_col : str or None
        If None, datetime is taken from the index.
    groupby : {"month", "dayofyear", "hour", "overall"}
    y_as_hhmm : bool
        Format y axis as HH:MM (useful only if values are decimal hours).
    """

    dfi = df.copy()

    if datetime_col is not None:
        dfi["_datetime"] = pd.to_datetime(dfi[datetime_col])
    else:
        dfi["_datetime"] = pd.to_datetime(dfi.index)

    dfi = dfi.sort_values("_datetime")
    dfi = dfi.dropna(subset=[value_col])

    if len(dfi) == 0:
        raise ValueError(f"No valid data found in column '{value_col}'.")

    dfi["month"] = dfi["_datetime"].dt.month
    dfi["dayofyear"] = dfi["_datetime"].dt.dayofyear
    dfi["hour"] = dfi["_datetime"].dt.hour

    if groupby == "month":
        grouped = dfi.groupby("month")[value_col]
        positions = sorted(dfi["month"].unique())
        labels = [pd.Timestamp(2000, m, 1).strftime("%b") for m in positions]

    elif groupby == "dayofyear":
        grouped = dfi.groupby("dayofyear")[value_col]
        positions = sorted(dfi["dayofyear"].unique())
        labels = [str(p) for p in positions]

    elif groupby == "hour":
        grouped = dfi.groupby("hour")[value_col]
        positions = sorted(dfi["hour"].unique())
        labels = [f"{h:02d}:00" for h in positions]

    elif groupby == "overall":
        positions = [1]
        labels = ["All"]
        groups = [dfi[value_col].values]

    else:
        raise ValueError("groupby must be one of: 'month', 'dayofyear', 'hour', 'overall'.")

    if groupby != "overall":
        group_dict = {k: g.values for k, g in grouped}
        groups = [group_dict[p] for p in positions if p in group_dict and len(group_dict[p]) > 0]
        positions = [p for p in positions if p in group_dict and len(group_dict[p]) > 0]
        labels = labels[:len(positions)]

    if len(groups) == 0:
        raise ValueError("No non-empty groups to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.violinplot(
        groups,
        positions=positions,
        showmeans=showmeans,
        showmedians=showmedians,
        showextrema=showextrema,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45 if groupby in ("month", "dayofyear", "hour") else 0)

    ax.set_ylabel(ylabel if ylabel is not None else value_col)
    ax.set_title(title if title is not None else f"Distribution of {value_col}")
    ax.grid(axis="y", alpha=0.3)

    if y_as_hhmm:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(decimal_hours_to_hhmm))

    plt.tight_layout()
    return ax

def normalize_signal_by_column(series, coord, n_points=256):
    """
    Interpolate a signal on a normalized grid defined by another variable.

    Parameters
    ----------
    series : pandas.Series
        Signal values (e.g. shading), indexed like coord.
    coord : pandas.Series or array-like
        Monotonic coordinate used for normalization (e.g. azimuth).
    n_points : int
        Number of points in the normalized signal.

    Returns
    -------
    pandas.Series
        Signal interpolated on a normalized coordinate in [0, 1].
    """
    s = pd.Series(series).copy()
    c = pd.Series(coord).copy()

    df = pd.DataFrame({"signal": s, "coord": c}).dropna()
    df = df.sort_values("coord")

    # remove duplicate coord values if needed
    df = df.groupby("coord", as_index=False)["signal"].mean()

    if len(df) < 2:
        return None

    cmin = df["coord"].min()
    cmax = df["coord"].max()

    if cmax <= cmin:
        return None

    u_obs = (df["coord"] - cmin) / (cmax - cmin)
    u_new = np.linspace(0, 1, n_points)
    y_new = np.interp(u_new, u_obs, df["signal"])

    return pd.Series(y_new, index=u_new)

def plot_diurnal_heatmap(meteo, propname='ghi', cmap='viridis', figsize=(14, 8)):
    """
    Plot a meteorological daymap (days x hour of day).

    Parameters
    ----------
    meteo : pandas.DataFrame
        DataFrame with a DatetimeIndex.
    propname : str
        Name of the meteorological variable to plot.
    cmap : str
        Matplotlib colormap.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    # Ensure datetime index
    meteo = meteo.copy()
    meteo.index = pd.to_datetime(meteo.index)

    # Build day/hour table
    tmp = pd.DataFrame({
        "date": meteo.index.date,
        "hour": (
            meteo.index.hour
            + meteo.index.minute / 60
            + meteo.index.second / 3600
        ),
        propname: meteo[propname]
    })

    # Pivot: rows = days, columns = hours
    heatmap = tmp.pivot(
        index="date",
        columns="hour",
        values=propname
    )

    # Sort
    heatmap = heatmap.sort_index(axis=1)

    # Transpose: rows = hours, columns = days
    Z = heatmap.T.values

    # Mask missing values so they appear white
    Z = np.ma.masked_invalid(Z)

    # Real hour coordinates
    hours = heatmap.columns.values

    # Cell edges for hours
    if len(hours) > 1:
        dh = np.median(np.diff(hours)) / 2
    else:
        dh = 0.05

    y_edges = np.concatenate((
        [hours[0] - dh],
        hours[:-1] + dh,
        [hours[-1] + dh]
    ))

    # Cell edges for days
    x_edges = np.arange(len(heatmap.index) + 1)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        Z,
        shading='auto',
        cmap=cmap
    )

    # Missing values in white
    pcm.cmap.set_bad('white')

    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(propname)

    # Y axis
    ax.set_ylim(0, 24)
    ax.set_ylabel("Hour of day")
    ax.set_yticks(np.arange(0, 25, 3))
    ax.set_yticklabels(
        [f"{h:02d}:00" for h in range(0, 25, 3)]
    )

    # X axis dates
    n_days = len(heatmap.index)
    n_ticks = min(10, n_days)

    xticks = np.linspace(0, n_days - 1, n_ticks, dtype=int)

    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels(
        [
            pd.Timestamp(heatmap.index[i]).strftime("%Y-%m-%d")
            for i in xticks
        ],
        rotation=45,
        ha="right"
    )

    ax.set_xlabel("Date")
    ax.set_title(f"Daymap of {propname}")

    plt.tight_layout()

    return fig, ax
