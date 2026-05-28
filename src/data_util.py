import pandas
from matplotlib import pyplot as plt
import pytz
import datetime
import numpy as np
from openalea.plantgl.light.lightmanager import to_clockwise


def format_values(irradiances):
    from generateplot import id2position, IDDECAL
    rows = {}
    cols = {}
    for id, irr  in irradiances.iterrows():
        if id >= IDDECAL or id == 0:
            cols[id], rows[id] = id2position(id)
    irradiances = pandas.DataFrame({'row':rows, 'column':cols, 'irradiance':irradiances['irradiance'][(irradiances.index>= IDDECAL) | (irradiances.index == 0)]})
    irradiances = irradiances.sort_values(['column','row'])
    return irradiances

def matrix_values(df, property='irradiance', size = None):
    import numpy as np
    if size is None:
        size = (max(df['row'])+1,max(df['column'])+1)
    res = np.zeros(size)
    res.fill(np.nan)
    for index, row in df.iterrows():
        res[int(row['row']),int(row['column'])] = float(row[property])
    return res


def toarray(coord, values, size = None, pixsize = 5, pixlength = 0.5, boxes  = None):
    import numpy as np
    if size is None:
        size = (max(j for i,j in coord)+1,max(i for i,j in coord)+1)
    res = np.zeros([pixsize*i for i in size])
    res.fill(np.nan)
    for coordij, valueij in zip(coord, values):
        for i in range(pixsize):
            for j in range(pixsize):
                res[pixsize*coordij[1]+i, pixsize*coordij[0]+j] = float(valueij)

    if boxes  is not None:
        for box in boxes :
            for coordi, coordj in zip(box,box[1:]+box[:1]):                
                if coordi[1] == coordj[1]:
                    startj = round(min(coordi[0], coordj[0])//(pixlength/pixsize))
                    endj = round(max(coordi[0], coordj[0])//(pixlength/pixsize))
                    iconstant = round(coordi[1]//(pixlength/pixsize))
                    for j in range(int(startj), int(endj+1)):
                        res[iconstant, j] = np.nan
                elif coordi[0] == coordj[0]:
                    starti = round(min(coordi[1], coordj[1])//(pixlength/pixsize))
                    endi = round(max(coordi[1], coordj[1])//(pixlength/pixsize))
                    jconstant = round(coordi[0]//(pixlength/pixsize))
                    for i in range(int(starti), int(endi+1)):
                        res[i, jconstant] = np.nan    
    return res


def toarray(coord, values, size=None, pixsize=5, pixlength=0.5, boxes=None):
    coord = np.asarray(coord)
    values = np.asarray(values, dtype=float)

    if size is None:
        size = (coord[:,1].max() + 1, coord[:,0].max() + 1)

    res = np.full((size[0]*pixsize, size[1]*pixsize), np.nan)

    # remplissage des pixels
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

            # segments horizontaux
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

            # segments verticaux
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

import numpy as np


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
    # X in image --> correspond to second indice
    # Y in image  --> first indice
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

    # pixels marqués
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

    # --- DATA POUR VIOLIN ---
    if data is None:
        data = array.ravel()

    data = np.asarray(data)
    data = data[np.isfinite(data)]  # 🔥 crucial

    if len(data) == 0:
        print("Warning: no valid data for violin plot")

    # --- VIOLIN PLOT ---
    parts = axes[1].violinplot(
        data,
        positions=[1],   # mieux centré
        showmeans=True,
        showextrema=True,
        showmedians=True
    )

    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["Values"])
    axes[1].set_title("Distribution")

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()       


def plot_meteo(meteo,  property='c2shading', polar = True, cmap='jet', marker='o', blocking = True, sinprojection = False, colorbar = True, size = (8,8)):
    import matplotlib.pyplot as plt
    from openalea.plantgl.light.utils import plot_sky

    vmin, vmax = None, None
    if isinstance(property, str):
        if property == 'date_time':
            property_values = meteo.index.astype(int) / 1e9  # convert to timestamp
        elif property not in meteo.columns:
            raise ValueError('Unknown meteo property '+property)
        else:
            property_values = meteo[property]
        if property in ['c2shading','c3shading','c2shaded','c3shaded']:
            vmin, vmax = 0, 1
    else:
        property_values = property

    fig, ax = plot_sky(meteo['azimuth'], meteo['elevation'], property_values, cmap=cmap, background=None, bgresolution=1, 
             representation = 'polar' if polar else 'vector', projection ='sin' if sinprojection else 'flat', colorbarlabel = property if type(property) == str else (property.name if hasattr(property,'name') else ''), elevationticks = not polar, pointsize = 3, edgecolors= None,
             marker = None, vmin = vmin, vmax = vmax, colorbar = colorbar, xylabel= False)
    fig.set_size_inches(*size)
    plt.grid(False)
    #plt.show(block = blocking)

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


def plot_sensor_meteo(saving = True, blocking = False, polar = True, output = 'dataview', suffix = ''):
    import matplotlib.pyplot as plt
    import os
    from meteo import meteo, clearsky, clearskyQ, meteo200, clearsky200, clearskyQ200
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    figsize = (5,4)
    for sensor in ['c2','c3']:
        for _suffix, (_meteo, _clearsky, _clearskyQ) in zip(['','_200'], [(meteo, clearsky, clearskyQ), (meteo200, clearsky200, clearskyQ200)]):
            lsuffix = _suffix + '_' + suffix
            plot_meteo(_meteo,'ghi', blocking=blocking, polar=polar, size=figsize)
            if saving:
                plt.savefig(os.path.join(output,sensor+'_A_meteo'+lsuffix+'.png'))
                plt.close()
            
            plot_meteo(_meteo,(_meteo.index - pandas.Timestamp(_meteo.index[0].strftime('%Y/%m/%d 0:0:0'), tz=_meteo.index.tz)).total_seconds() % (24*3600), blocking=blocking, polar=polar, size=figsize)
            if saving:
                plt.savefig(os.path.join(output,sensor+'_A_time'+lsuffix+'.png'))
                plt.close()
            
            plot_meteo(_meteo, sensor, blocking=blocking, polar=polar, size=figsize)
            if saving:
                plt.savefig(os.path.join(output,sensor+'_B_sensor'+lsuffix+'.png'))
                plt.close()

            plot_meteo(_meteo,sensor+'shading', blocking=blocking, polar=polar, size=figsize)
            if saving:
                plt.savefig(os.path.join(output,sensor+'_C_shading'+lsuffix+'.png'))
                plt.close()

            plot_meteo(_clearsky,sensor+'shading', blocking=blocking, polar=polar, size=figsize)
            if saving:
                plt.savefig(os.path.join(output,sensor+'_D_clearsky_shading'+lsuffix+'.png'))
                plt.close()

            plot_meteo(_clearsky,sensor+'shaded', blocking=blocking, polar=polar, cmap='binary', size=figsize, colorbar=False)
            if saving:
                plt.savefig(os.path.join(output,sensor+'_E_clearsky_shaded'+lsuffix+'.png'))
                plt.close()

            plot_meteo(_clearskyQ,sensor+'shaded', blocking=blocking, polar=polar, cmap='binary', size=figsize, colorbar=False)
            if saving:
                plt.savefig(os.path.join(output,sensor+'_F_quantized_clearsky_shaded'+lsuffix+'.png'))
                plt.close()

def filter_coord(coords, decal = -4):
    result = []
    for col, row in coords:
        if col >= (23+decal) and col <= (77-decal) and col != (36+decal) and col != (66+decal) and row >=9 and row <= 26 :
             result.append((col, row))
    return result

def filter_dict(df):
    validcoords = set(filter_coord(df.keys()))
    return {k:v for k,v in df.items() if k in validcoords}

def filter_dataframe(df):
    validcoords = set(filter_coord(df.keys()))
    return df[df.apply(lambda row: (row['column'], row['row']) in validcoords, axis=1)]

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

def load_irradiances(path = 'result/weather2023', compression = True):
    import os, time
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

    for p in sorted(os.listdir(path)):
        if p.startswith('simulation_') and p.endswith('.csv'):
            print('Loading irradiance data from', p)
            height, date = retrieve_info_from_fname(p)
            df = read_csv(os.path.join(path, p), index_col=0, sep='\t')
            df['date_time'] = date
            df['height'] = height
            results.append(df)
    print('Total time to load irradiance data from', repr(path), ':', time.time() - t)
    results = concat(results, ignore_index=True)
    if results['height'].nunique() == 1:
        results.attrs['height'] = results['height'].iloc[0]
        results = results.drop(columns=['height'])
    if compression:
        results.to_pickle(gfile)
        irradiances_cache[gfile] = results
    return results

def compress_irradiances(path = 'result/weather2023'):
    import os, time
    from pandas import read_csv, concat, read_pickle
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
def get_irradiances_per_plot(path = 'result/weather2023'):
    if path in irradiances_per_plot_cache:
        print('Loading irradiance per plot data from cache for path', repr(path))
        return irradiances_per_plot_cache[path]
    df = load_irradiances(path)
    result = retrieve_irradiance_per_plot(df)
    irradiances_per_plot_cache[path] = result
    return result

def get_clusters(fname = 'clusters.csv', center = True):
    import pandas as pd
    data = pd.read_csv(fname)
    data['selected'] = data['Selected'] == 'selected'
    clusters = {}
    for i, row in data.iterrows():
        col, row, group, selected = row['Col'],row['Row'],row['group_clearsky'],row['selected']
        if not center or selected:
            clusters[(col, row)] = group
    return clusters


def sort_irradiances_per_cluster(irradiances_per_plot, clusters):
    result = {}
    for (col, row), df in irradiances_per_plot.items():
        group = clusters.get((col, row), None)
        if group is not None:
            if group not in result:
                result[group] = {}
            result[group][(col, row)] = df
    return result

def sort_irradiances_per_cluster_from_file(irradiances_per_plot, fname = 'data/clusters.csv', center = True):
    clusters = get_clusters(fname, center)
    return sort_irradiances_per_cluster(irradiances_per_plot, clusters)

def get_irradiances_per_cluster(path = 'result/weather2023', fname = 'data/clusters.csv', center = True):
    return sort_irradiances_per_cluster(get_irradiances_per_plot(path), fname, center)


if __name__ == '__main__':
    #print(len(clearsky))
    #simplerclearsky = quantize_meteo(clearsky,1)
    #print(len(simplerclearsky))
    #plot_meteo(clearsky1, "c3shading",cmap='jet')
    #m = meteo_subsampling(meteo,1, seed=1)
    #meteo[['ghi','dhi','c2','c3']].plot()
    plot_sensor_meteo(blocking=False)
