"""Simulation data loading, date operations, and grid maps."""

import datetime

import numpy as np
import pandas
from matplotlib import pyplot as plt


def matrix_values(df, property='irradiance', size = None):
    import numpy as np
    if size is None:
        size = (max(df['row'])+1,max(df['column'])+1)
    res = np.zeros(size)
    res.fill(np.nan)
    for index, row in df.iterrows():
        res[int(row['row']),int(row['column'])] = float(row[property])
    return res


def toarray(coord, values, size=None, pixsize=5, pixlength=0.5, boxes=None):
    coord = np.asarray(coord)
    values = np.asarray(values, dtype=float)

    if size is None:
        size = (coord[:,1].max() + 1, coord[:,0].max() + 1)

    res = np.full((size[0]*pixsize, size[1]*pixsize), np.nan)

    # Fill pixels
    base_i = coord[:,1] * pixsize
    base_j = coord[:,0] * pixsize

    di, dj = np.meshgrid(np.arange(pixsize), np.arange(pixsize), indexing='ij')

    I = base_i[:, None, None] + di
    J = base_j[:, None, None] + dj

    res[I, J] = values[:, None, None]

    # boxes
    if boxes is not None:
        scale = pixlength / pixsize

        for box in boxes:
            box = np.asarray(box)
            p1 = box
            p2 = np.roll(box, -1, axis=0)

            # Horizontal segments
            mask_h = p1[:,1] == p2[:,1]
            if np.any(mask_h):
                y = np.round(p1[mask_h,1] / scale).astype(int)
                x1 = np.round(np.minimum(p1[mask_h,0], p2[mask_h,0]) / scale).astype(int)
                x2 = np.round(np.maximum(p1[mask_h,0], p2[mask_h,0]) / scale).astype(int)

                mask_in_plot = (y >= 0) & (y < res.shape[0])

                y = y[mask_in_plot]
                x1 = np.clip(x1, 0, res.shape[1]-1)[mask_in_plot]
                x2 = np.clip(x2, 0, res.shape[1]-1)[mask_in_plot]

                for yi, xa, xb in zip(y, x1, x2):
                    res[yi, xa:xb+1] = np.nan

            # Vertical segments
            mask_v = p1[:,0] == p2[:,0]
            if np.any(mask_v):
                x = np.round(p1[mask_v,0] / scale).astype(int)
                y1 = np.round(np.minimum(p1[mask_v,1], p2[mask_v,1]) / scale).astype(int)
                y2 = np.round(np.maximum(p1[mask_v,1], p2[mask_v,1]) / scale).astype(int)

                mask_in_plot = (x >= 0) & (x < res.shape[1])

                x = x[mask_in_plot]
                y1 = np.clip(y1, 0, res.shape[0]-1)[mask_in_plot]
                y2 = np.clip(y2, 0, res.shape[0]-1)[mask_in_plot]

                for xi, ya, yb in zip(x, y1, y2):
                    res[ya:yb+1, xi] = np.nan

    return res


def dataframe_toarray(df, coords, column, size=None, pixsize=5, pixlength=0.5, boxes=None):
    """
    Convert a dataframe of per-plot values into a 2D pixel array.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - 'plot' : plot index
        - column  : value to display
    coords : array-like of shape (n_plots, 2)
        coords[p] = (x, y) position of plot p
    column : str
        Name of the dataframe column to display
    size : tuple or None
        Grid size as (ny, nx). If None, inferred from coords actually used
    pixsize : int
        Number of pixels per plot cell
    pixlength : float
        Physical plot size
    boxes : list or None
        Optional list of polygons to overlay as NaN lines

    Returns
    -------
    res : 2D ndarray
        Pixel array
    """

    coords = np.asarray(coords)

    if "plot" not in df.columns:
        raise ValueError("DataFrame must contain a 'plot' column.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # --- keep only valid plots
    plots = df["plot"].to_numpy(dtype=int)
    values = df[column].to_numpy(dtype=float)

    valid = (plots >= 0) & (plots < len(coords))
    plots = plots[valid]
    values = values[valid]

    coord = coords[plots]

    if coord.size == 0:
        raise ValueError("No valid plot coordinates found.")

    # infer size from used coordinates
    if size is None:
        size = (coord[:, 1].max() + 1, coord[:, 0].max() + 1)

    res = np.full((size[0] * pixsize, size[1] * pixsize), np.nan)

    # --- fill pixels (same logic as your function)
    base_i = coord[:, 1].astype(int) * pixsize
    base_j = coord[:, 0].astype(int) * pixsize

    di, dj = np.meshgrid(np.arange(pixsize), np.arange(pixsize), indexing='ij')

    I = base_i[:, None, None] + di
    J = base_j[:, None, None] + dj

    res[I, J] = values[:, None, None]

    # --- boxes
    if boxes is not None:
        scale = pixlength / pixsize

        for box in boxes:
            box = np.asarray(box)
            p1 = box
            p2 = np.roll(box, -1, axis=0)

            # horizontal segments
            mask_h = p1[:, 1] == p2[:, 1]
            if np.any(mask_h):
                y = np.round(p1[mask_h, 1] / scale).astype(int)
                x1 = np.round(np.minimum(p1[mask_h, 0], p2[mask_h, 0]) / scale).astype(int)
                x2 = np.round(np.maximum(p1[mask_h, 0], p2[mask_h, 0]) / scale).astype(int)

                mask_in_plot = (y >= 0) & (y < res.shape[0])

                y = y[mask_in_plot]
                x1 = np.clip(x1, 0, res.shape[1] - 1)[mask_in_plot]
                x2 = np.clip(x2, 0, res.shape[1] - 1)[mask_in_plot]

                for yi, xa, xb in zip(y, x1, x2):
                    res[yi, xa:xb + 1] = np.nan

            # vertical segments
            mask_v = p1[:, 0] == p2[:, 0]
            if np.any(mask_v):
                x = np.round(p1[mask_v, 0] / scale).astype(int)
                y1 = np.round(np.minimum(p1[mask_v, 1], p2[mask_v, 1]) / scale).astype(int)
                y2 = np.round(np.maximum(p1[mask_v, 1], p2[mask_v, 1]) / scale).astype(int)

                mask_in_plot = (x >= 0) & (x < res.shape[1])

                x = x[mask_in_plot]
                y1 = np.clip(y1, 0, res.shape[0] - 1)[mask_in_plot]
                y2 = np.clip(y2, 0, res.shape[0] - 1)[mask_in_plot]

                for xi, ya, yb in zip(x, y1, y2):
                    res[ya:yb + 1, xi] = np.nan

    return res

def toimage(array, fname='out.png', vmin=None, vmax=None, xticklabels= None, yticklabels = None, markpixel= None, cmap='jet',):
    # X in image --> correspond to second index
    # Y in image  --> first index
    plt.matshow(array, cmap='jet',vmin = vmin, vmax=vmax, origin='lower')
    plt.colorbar()
    if markpixel:
        plt.scatter([m[0] for m in markpixel],[m[1] for m in markpixel], c=[m[2] if len(m) > 2 else 'white' for m in markpixel ])
    if  xticklabels:
        nbticks = len(xticklabels)
        xrange = array.shape[0]
        plt.yticks(np.arange(0, xrange+1e-3, xrange/(nbticks-1)), labels=xticklabels)
    if yticklabels:
        nbticks = len(yticklabels)
        yrange = array.shape[1]
        plt.xticks(np.arange(0, yrange+1e-3, yrange/(nbticks-1)), labels=yticklabels, rotation=60)
    #plt.tight_layout()
    #plt.show(block=False)
    plt.savefig(fname)
    plt.close()



def toimage_with_violin(
    array,
    data=None,
    fname='out.png',
    vmin=None,
    vmax=None,
    xticklabels=None,
    yticklabels=None,
    markpixel=None,
    cmap='jet',
):
    fig, axes = plt.subplots(1, 2, figsize=(21, 5),
        gridspec_kw={'width_ratios': [4, 1]})

    # --- IMAGE ---
    img = axes[0].imshow(array, cmap=cmap, vmin=vmin, vmax=vmax,
                         origin='lower', aspect='auto')
    fig.colorbar(img, ax=axes[0])

    # Marked pixels
    if markpixel is not None:
        axes[0].scatter(
            [m[0] for m in markpixel],
            [m[1] for m in markpixel],
            c=[m[2] if len(m) > 2 else 'white' for m in markpixel]
        )

    # ticks Y
    if xticklabels is not None:
        nbticks = len(xticklabels)
        xrange = array.shape[0]
        ticks = np.linspace(0, xrange - 1, nbticks)
        axes[0].set_yticks(ticks)
        axes[0].set_yticklabels(xticklabels)

    # ticks X
    if yticklabels is not None:
        nbticks = len(yticklabels)
        yrange = array.shape[1]
        ticks = np.linspace(0, yrange - 1, nbticks)
        axes[0].set_xticks(ticks)
        axes[0].set_xticklabels(yticklabels, rotation=60)

    # --- VIOLIN PLOT DATA ---
    if data is None:
        data = array.ravel()

    data = np.asarray(data)
    data = data[np.isfinite(data)]  # 🔥 crucial

    if data.size:
        axes[1].violinplot(
            data, positions=[1], showmeans=True, showextrema=True, showmedians=True,
        )

    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["Values"])
    axes[1].set_title("Distribution")

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def get_meteo_days(meteo):
    return np.unique(meteo.index.date)

def meteo_select_dates(meteo, selected_dates):
    if type(selected_dates) in [str,list]:
        if type(selected_dates) == str :
            if ',' in selected_dates:
                selected_dates = selected_dates.split(',')
            else:
                selected_dates = [selected_dates]
        if type(selected_dates[0]) == str :
            selected_dates = [datetime.datetime.strptime(date, "%Y/%m/%d").date() for date in selected_dates]
    mask = np.isin(meteo.index.date, selected_dates)
    selected = meteo[mask]
    return selected

def timeline_select_dates(timeline, selected_dates):
    if type(selected_dates) in [str,list]:
        if type(selected_dates) == str :
            if ',' in selected_dates:
                selected_dates = selected_dates.split(',')
            else:
                selected_dates = [selected_dates]
        if type(selected_dates[0]) == str :
            selected_dates = [datetime.datetime.strptime(date, "%Y/%m/%d").date() for date in selected_dates]
    mask = np.isin(timeline.date, selected_dates)
    selected = timeline[mask]
    return selected

def meteo_subsampling(meteo, nbdates = 50, seed = 0):
    dates = get_meteo_days(meteo)
    if nbdates > len(dates) : return meteo
    np.random.seed(seed)
    selected_dates = np.random.choice(dates, size=nbdates, replace=False)
    return meteo_select_dates(meteo, selected_dates)


irradiances_cache = {}

def retrieve_info_from_fname(fname):
    import re, datetime
    begin = 'simulation'
    end = '.csv'
    assert fname.startswith(begin) and fname.endswith(end), "Filename must start with '"+begin+"' and end with '"+end+"'"

    pattern = r'20(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})'
    match = re.search(pattern, fname)
    if match:
        year, month, day, hour, minute = match.groups()
        date = datetime.datetime(2000+int(year), int(month), int(day), int(hour), int(minute))
    else:
        raise ValueError("Filename does not contain a valid date and time.")
    pattern = r'_([\d_]+)_'
    match = re.search(pattern, fname[len(begin):match.start()], re.IGNORECASE)
    if match:
        height = float(match.groups()[0].replace('_', '.'))
    else:
        raise ValueError("Filename does not contain a valid height value.")
    return height, date

def load_irradiances(path, compression = True):
    import os, time
    from pathlib import Path
    from tempfile import NamedTemporaryFile
    from pandas import read_csv, concat, read_pickle
    results = []
    t = time.time()
    basename = 'global_irradiances.pkl'
    gfile = os.path.join(path, basename)
    if os.path.exists(gfile+'.zip'):
        gfile = gfile+'.zip'

    if gfile in irradiances_cache:
        result = irradiances_cache[gfile]
        print('Loading global irradiance data from cache in',(time.time() - t))
        return result

    if os.path.exists(gfile):
        result = read_pickle(gfile)
        print('Loading global irradiance data from', repr(gfile),'in', time.time() - t, 'seconds')
        irradiances_cache[gfile] = result
        return result

    if not os.path.isdir(path):
        raise FileNotFoundError(f'Simulation directory not found at {path}')
    for p in sorted(os.listdir(path)):
        if p.startswith('simulation_') and p.endswith('.csv'):
            print('Loading irradiance data from', p)
            height, date = retrieve_info_from_fname(p)
            df = read_csv(os.path.join(path, p), index_col=0, sep='\t')
            df['date_time'] = date
            df['height'] = height
            results.append(df)
    print('Total time to load irradiance data from', repr(path), ':', time.time() - t)
    if not results:
        raise FileNotFoundError(f'No simulation results found in {path}')
    results = concat(results, ignore_index=True)
    if results['height'].nunique() == 1:
        results.attrs['height'] = results['height'].iloc[0]
        results = results.drop(columns=['height'])
    if compression:
        with NamedTemporaryFile(dir=path, prefix=".irradiance-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
        try:
            results.to_pickle(temporary)
            temporary.replace(gfile)
        finally:
            temporary.unlink(missing_ok=True)
        irradiances_cache[gfile] = results
    return results

def compress_irradiances(path):
    import os, time
    t = time.time()
    gfile = os.path.join(path, 'global_irradiances.pkl')
    if os.path.exists(gfile):
        print('Global irradiance data already compressed in', repr(gfile))
        return
    results = load_irradiances(path, compression = True)
    if os.path.exists(gfile):
        for p in sorted(os.listdir(path)):
            if p.startswith('simulation_') and p.endswith('.csv'):
                os.remove(os.path.join(path, p))
    print('Total time to compress irradiance data from', repr(path), ':', time.time() - t, 'seconds')
    return gfile


def retrieve_irradiance_from_plot(dataframe, row, column):
    df =  dataframe.loc[(dataframe['row'] == row) & (dataframe['column'] == column)]
    df.set_index('date_time', inplace=True)
    return df

def retrieve_irradiance_per_plot(dataframe):
    import time
    t = time.time()
    result = {}
    for (i,j), df in dataframe.groupby(['column','row']):
        result[(i,j)] = df.reset_index().set_index('date_time')
    print('Total time to retrieve irradiance from all plots:', time.time() - t)
    return result

irradiances_per_plot_cache = {}
def get_irradiances_per_plot(path):
    if path in irradiances_per_plot_cache:
        print('Loading irradiance per plot data from cache for path', repr(path))
        return irradiances_per_plot_cache[path]
    df = load_irradiances(path, compression=False)
    result = retrieve_irradiance_per_plot(df)
    irradiances_per_plot_cache[path] = result
    return result

def sort_irradiances_per_cluster(irradiances_per_plot, clusters):
    result = {}
    for (col, row), df in irradiances_per_plot.items():
        group = clusters.get((col, row), None)
        if group is not None:
            if group not in result:
                result[group] = {}
            result[group][(col, row)] = df
    return result
