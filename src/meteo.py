
import pandas
from localization import *


date2m = '2023/6/21 12:0:0'


def read_meteo(data_file='2023_PAR_c1.txt', localisation = localisation['timezone']):
    """ reader for mango meteo files """
    data = pandas.read_csv(data_file, delimiter = '\t', dayfirst=True)

    #data = data.rename(columns={'date_time':'date_time',
    #                             'par':'global_radiation'})
    # convert kW.m-2 to W.m-2
    #data['global_radiation'] *= 1000.
    index = pandas.DatetimeIndex(data['date_time']).tz_localize(localisation)
    data.drop(columns=["date_time"], inplace=True)
    data = data.set_index(index)
    return data


def get_meteo(capteur='c1'):
    ppfd = read_meteo(data_file='2023_PAR_'+capteur+'.txt')
    global_radiation = read_meteo(data_file='2023_globalradiations_'+capteur+'.txt')
    #ppfd.rename(columns={'par':'PAR'}, inplace=True)
    from openalea.astk.sky_irradiance import sky_irradiance
    sky_irr = sky_irradiance(dates=ppfd.index, ghi=global_radiation['Rg'], ppfd=ppfd['PAR'], latitude=localisation['latitude'], longitude=localisation['longitude'], altitude=localisation['altitude'])
    return sky_irr

def prepare_meteo(shadinglevel=0.5):
    c1 = get_meteo('c1')
    c2 = read_meteo('2023_globalradiations_c2.txt')
    c3 = read_meteo('2023_globalradiations_c3.txt')
    c1 = c1.loc[c1.index.isin(c2.index)]
    c1 = c1.loc[c1.index.isin(c3.index)]
    c2 = c2.loc[c2.index.isin(c1.index)]
    c3 = c3.loc[c3.index.isin(c1.index)]

    r = c2['Rg'].gt(c1['ghi'])
    c2.mask(r,c1[r]['ghi'], axis=1, inplace=True)
    r = c3['Rg'].gt(c1['ghi'])
    c3.mask(r,c1[r]['ghi'], axis=1, inplace=True)

    meteo = c1
    meteo = meteo.join(c2['Rg'], how='inner')
    meteo.rename(columns={'Rg':'c2'}, inplace=True)
    meteo = meteo.join(c3['Rg'], how='inner')
    meteo.rename(columns={'Rg':'c3'}, inplace=True)

    #meteo['azimuth'] = to_clockwise( meteo['azimuth'])

    # compute shading ratios
    meteo['c2shading'] = 1- meteo['c2']/meteo['ghi']
    meteo['c3shading'] = 1- meteo['c3']/meteo['ghi']

    # Estimate shading presence with a threshold on shading ratio
    meteo['c2shaded'] = meteo['c2shading'] > shadinglevel
    meteo['c3shaded'] = meteo['c3shading'] > shadinglevel

    # on enleve le premier et le dernier jour qui sont incomplets
    meteo = meteo.loc[
    (meteo.index.date != meteo.index.date[0]) &
    (meteo.index.date != meteo.index.date[-1])]

    return meteo

def quantize_meteo(meteo, precision = 1):
    collector = {}
    for date, row in meteo.iterrows():
        azimuth, elevation = row['azimuth'], row['elevation']
        qaz = round(azimuth/precision)*precision
        qel = round(elevation/precision)*precision
        collector.setdefault((qaz, qel), [])
        collector[(qaz,qel)].append((date,row))
    propname = ['ghi','dhi','c2','c3','c2shading','c3shading','c2shaded','c3shaded']
    res = { name : [] for name in ['azimuth', 'elevation','zenith','date_time']+propname }
    for (az,el), rows in collector.items():
        res['azimuth'].append(az)
        res['elevation'].append(el)
        res['zenith'].append(90-el)
        imax = max([(row['ghi'],i) for i,(date, row) in enumerate(rows)], key=lambda x : x[0])[1]
        res['date_time'].append(rows[imax][0])
        rowmax = rows[imax][1]
        for pname in propname:
            res[pname].append(rowmax[pname])
    data = pandas.DataFrame(res)
    index = pandas.DatetimeIndex(data['date_time'])
    data.drop(columns=["date_time"], inplace=True)
    data = data.set_index(index)
    return data
 
meteo, meteo90, meteo200, clearsky, clearsky90, clearsky200, clearskyQ, clearskyQ90, clearskyQ200 = [None]*9

def setup_meteo():
    global meteo, meteo90, meteo200, clearsky, clearsky90, clearsky200, clearskyQ, clearskyQ90, clearskyQ200
    if meteo is None: 
        print("Preparing meteo data...")
        meteo = prepare_meteo()
        meteo90 = meteo[meteo.index < date2m]
        meteo200 = meteo[meteo.index >= date2m]
        clearsky = meteo[meteo['ghi'] *0.5 > meteo['dhi']]
        clearsky90 = meteo90[meteo90['ghi'] *0.5 > meteo90['dhi']]
        clearsky200 = meteo200[meteo200['ghi'] *0.5 > meteo200['dhi']]
        clearskyQ = quantize_meteo(clearsky,1)
        clearskyQ90 = quantize_meteo(clearsky90,1)
        clearskyQ200 = quantize_meteo(clearsky200,1)

def generate_meteo(attenuation = None):
    dates = meteo.index
    from openalea.astk.sky_irradiance import sky_irradiance
    sky_irr = sky_irradiance(dates=dates, attenuation=attenuation, latitude=localisation['latitude'], longitude=localisation['longitude'], altitude=localisation['altitude'])
    return sky_irr

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
    if m == 60:  # correction arrondi
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

    # ✨ format HH:MM sur axe Y
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
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_decimal_hours_to_hhmm))

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

setup_meteo()

if __name__ == "__main__":
    daylength, ax = violin_daylength_from_df(meteo, groupby="overall")
    print(daylength.describe())
    plt.show()
    violinplot_from_df(
    meteo,
    value_col="azimuth",
    groupby="month",
    ylabel="Solar azimuth (deg)",
    title="Distribution of solar azimuth by month"
    )
    plt.show()