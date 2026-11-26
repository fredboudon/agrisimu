"""Conventions de grille, capteurs et traitements du site."""

import pandas
from agrisimu.data_util import get_irradiances_per_plot, sort_irradiances_per_cluster

from .paths import DATA_DIR, OUTPUT_DIR


def format_values(irradiances):
    from .generateplot import id2position, IDDECAL
    rows = {}
    cols = {}
    for id, irr  in irradiances.iterrows():
        if id >= IDDECAL or id == 0:
            cols[id], rows[id] = id2position(id)
    irradiances = pandas.DataFrame({'row':rows, 'column':cols, 'irradiance':irradiances['irradiance'][(irradiances.index>= IDDECAL) | (irradiances.index == 0)]})
    irradiances = irradiances.sort_values(['column','row'])
    return irradiances

def plot_meteo(meteo,  property='c2shading', polar = True, cmap='jet', marker='o', blocking = True, sinprojection = False, colorbar = True, size = (8,8)):
    import matplotlib.pyplot as plt
    from openalea.plantgl.light.utils import plot_sky

    vmin, vmax = None, None
    if isinstance(property, str):
        if property == 'date_time':
            epoch = pandas.Timestamp('1970-01-01', tz='UTC' if meteo.index.tz is not None else None)
            property_values = (meteo.index - epoch).total_seconds()
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

def plot_sensor_meteo(saving = True, blocking = False, polar = True, output = OUTPUT_DIR / 'dataview', suffix = ''):
    import matplotlib.pyplot as plt
    import os
    from .meteo import setup_meteo

    weather = setup_meteo()
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    figsize = (5,4)
    for sensor in ['c2','c3']:
        for _suffix, (_meteo, _clearsky, _clearskyQ) in zip(['','_200'], [(weather.meteo, weather.clearsky, weather.clearskyQ),
                                                            (weather.meteo200, weather.clearsky200, weather.clearskyQ200)]):
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
    validcoords = set(filter_coord(zip(df['column'], df['row'])))
    return df[df.apply(lambda row: (row['column'], row['row']) in validcoords, axis=1)]

def get_clusters(fname = DATA_DIR / 'clusters.csv', center = True):
    import pandas as pd
    data = pd.read_csv(fname)
    data['selected'] = data['Selected'] == 'selected'
    clusters = {}
    for i, row in data.iterrows():
        col, row, group, selected = row['Col'],row['Row'],row['group_clearsky'],row['selected']
        if not center or selected:
            clusters[(col, row)] = group
    return clusters


def sort_irradiances_per_cluster_from_file(irradiances_per_plot, fname = DATA_DIR / 'clusters.csv', center = True):
    clusters = get_clusters(fname, center)
    return sort_irradiances_per_cluster(irradiances_per_plot, clusters)

def get_irradiances_per_cluster(path = OUTPUT_DIR / "result" / "weather2023", fname = DATA_DIR / 'clusters.csv', center = True):
    return sort_irradiances_per_cluster_from_file(get_irradiances_per_plot(path), fname, center)



if __name__ == "__main__":
    plot_sensor_meteo()
