
import pandas
from openalea.plantgl.all import *

import os, sys

from importlib import reload
import generateplot; reload(generateplot)
from generateplot import *

import data_util; reload(data_util)
from data_util import *
from meteo import *

import time, datetime, pytz
from os.path import join

import matplotlib.pyplot as plt


DEBUG = False
RESOLUTION = 0.01  # in meters

#sensordict = {'c2': Vector3(35.43,0.19,2), 'c3': Vector3(34.64,2.89,2)}
sensordict = {'c2': Vector3(27.3,4.63,2), 'c3': Vector3(20.54,7.82,2)}

""" Ne considerez que du '17-May' au '31-Oct' """
def process_sensors(meteo=meteo, mindate = None, maxdate = None, view = False, outdir = 'result_sensors'):
    """
    Process light simulation for an agrivoltaic scene over a specified date range.
    Simulates solar irradiance and sky conditions based on meteorological data,
    computing light distribution across sensor geometry for each timestep.
    Args:
        mindate (date): Minimum date for simulation (default: date(5,1,0) - May 1st).
        maxdate (date): Maximum date for simulation (default: date(11,1,0) - November 1st).
        meteofile (str): Path to meteorological data file (default: 'weather.txt').
        sensorheight (float): Height of sensor above ground level in meters (default: 0.0).
        view (bool): Whether to save 3D scene representation (default: True).
        outdir (str): Output directory for results (default: 'result'). If None, results are returned in memory.
    Returns:
        list: If outdir is None, returns list of tuples (cdate, result) containing simulation results.
              If outdir is specified, returns empty list and saves results to CSV and image files.
    Notes:
        - Creates output directory if it does not exist.
        - Skips simulation for timesteps where results already exist.
        - Generates irradiance maps and sky maps as PNG images.
        - Computes transmitted irradiance as a fraction of global horizontal irradiance.
        - Only processes timesteps with positive global irradiance values.
    """
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    fname = 'sensor_simulation'
    if outdir:
        if not mindate is None:
             fname+='_'+str(mindate)
        if not maxdate is None:
             fname+='_'+str(maxdate)

        if os.path.exists(join(outdir,fname+'.csv')):
            print('Results already exist for the specified date range. Loading from file.')
            return pandas.read_csv(join(outdir,fname+'.csv'), sep='\t', index_col=0, parse_dates=True)

    if not mindate is None and maxdate is None:
         maxdate = mindate + datetime.timedelta(1)

    # an agrivoltaic scene (generate plot)
    scene = agristructure()


    l = LightEstimator(scene)
    l.localize(name = 'Camargue', **localisation)

    print('Set diffuse irradiance map as precomputed...')

    meteo90 = meteo[meteo.index < date2m]
    meteo200 = meteo[meteo.index >= date2m]

    meteos = [meteo90, meteo200]
    heights = [0.9, 2]

    results = {sensorid : [] for sensorid in sensordict.keys()}
    resultdate = []
    results['date_time'] = resultdate
    heightvalues = []
    results['height'] = heightvalues

    for _meteo, height in zip(meteos, heights):
      if len(_meteo.index) > 0:
        l.clear_sensors()        
        for name, pos in sensordict.items():
                pos.z = height
                l.add_sensor(name, pos)
                l.sensors[name].compute()
                if view:
                    plt.ion()
                    l.sensors[name].view()
                    if outdir:
                        plt.savefig(join(outdir,'sensor_'+name+'_'+str(height)+'_skymap.png'))
                    plt.close()


        t = time.time()
        for cdate, row in _meteo.iterrows():
            globalirr, diffuseirr, c2, c3 = row['ghi'],row['dhi'],row['c2'],row['c3']
            if (mindate is None or cdate >= mindate) and (maxdate is None or cdate < maxdate) and globalirr > 0:
                print(cdate, globalirr, diffuseirr)
                l.clear_lights()
                l.add_astk_sun_sky(dates = [cdate], ghi = globalirr, dhi = diffuseirr, sky_subdivision = 6)
                #fname = join(outdir,'simulation_'+str(sensorheight).replace('.','_')+'_'+cdate.strftime('%Y-%m-%d-%H-%M'))
                result_sensor = l.estimate_sensors()
                resultdate.append(cdate)
                heightvalues.append(height)
                for sid, value in result_sensor.items():
                    results[sid].append(value)

    print('  simulation time:', time.time()-t)

    results = pandas.DataFrame(results)
    index = pandas.DatetimeIndex(results['date_time'])
    results.drop(columns=["date_time"], inplace=True)
    results = results.set_index(index)
    results.rename(columns={sensorid : 'simulated_'+sensorid for sensorid in sensordict.keys()}, inplace=True)
    results = results.join(meteo.loc[results.index])
    
    if outdir:
        results.to_csv(join(outdir,fname+'.csv'),sep='\t')
        
    return results

def resample(data, target_length = '1h'):
    """Resample une série temporelle à une longueur cible en utilisant l'interpolation linéaire"""
    #firstprop = ['height', 'azimuth', 'zenith', 'elevation']
    meanprop = ['simulated_c2', 'simulated_c3',  'c2', 'c3']
    data = data[meanprop]
    agg = {}
    #agg.update({ prop : 'first' for prop in firstprop if prop in data.columns})
    agg.update({ prop : 'mean' for prop in meanprop if prop in data.columns})
    df = data.resample(target_length).agg(agg).dropna(subset=['simulated_c2'])
    for col in df.columns:
        if col  in ['index', 'row', 'column']:
            df[col] = df[col].astype(data[col].dtype)
    return df

def resample_sum(data, target_length = '1D'):
    """Resample une série temporelle à une longueur cible en utilisant l'interpolation linéaire"""
    sumprop = ['simulated_c2', 'simulated_c3',  'c2', 'c3']
    data = data[sumprop]
    agg = {}
    #agg.update({ prop : 'first' for prop in firstprop if prop in data.columns})
    agg.update({ prop : 'sum' for prop in sumprop if prop in data.columns})
    df = data.resample(target_length).agg(agg).dropna(subset=['simulated_c2'])
    for col in sumprop:
        df[col] = df[col] * 5 * 60 * 1e-6  # convertir de W/m² à MJ/m² en multipliant par le nombre de secondes dans la période (5min = 300s) et en divisant par 1e6 pour convertir de J à MJ
    for col in df.columns:
        if col  in ['index', 'row', 'column']:
            df[col] = df[col].astype(data[col].dtype)
    return df

def plot_comparison(results, ci, coloring = 'h', label = 'Global Radiation (W m$^{-2}$)'):
        from sklearn.metrics import r2_score
        from matplotlib.colors import BoundaryNorm

        df = results[[ci, 'simulated_' + ci]].dropna()

        x = df[ci]
        y = df['simulated_' + ci]

        # --- Metrics ---
        rmse = np.sqrt(((x - y) ** 2).mean())
        bias = (y - x).mean()

        # R² (coefficient de détermination)
        r2 = r2_score(x, y)

        plt.figure(figsize=(6,5))

        colorindex = df.index.hour if coloring == 'h' else ( df.index.month if coloring == 'M' else df.index.dayofyear)
        minh = min(colorindex)
        maxh = max(colorindex)

        cmap = plt.get_cmap('viridis', maxh-minh+1)  # 24 couleurs
        bounds = np.arange(minh-0.5, maxh+0.5, 1)   # bornes pour centrer sur entiers
        norm = BoundaryNorm(bounds, cmap.N)

        sc = plt.scatter(x, y,c=colorindex,         
                         cmap=cmap,
                         norm=norm, s=10, alpha=0.7)
        
        plt.xlabel(f'Observed {label}')
        plt.ylabel(f'Simulated {label}')
        plt.colorbar(sc, ticks=range(minh, maxh + 1), label=('Hour' if coloring == 'h' else ('Month' if coloring == 'M' else 'Day of Year')))

        xmin, xmax = x.min(), x.max()
        plt.plot([xmin, xmax], [xmin, xmax], 'k--')

                # --- Texte des métriques ---
        textstr = (
            f'RMSE = {rmse:.2f}\n'
            f'R² = {r2:.2f}\n'
            f'Bias = {bias:.2f}'
        )

        # Position en coordonnées axes (0-1)
        plt.text(
            0.05, 0.95, textstr,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

def plot_result(results, outdir = 'result_sensors'):
    import matplotlib.dates as mdates
    print(results.columns.tolist())
    figsize=(15,5)
    sensors = ['c2','c3']
    for ci in sensors:
        results[[ci,'simulated_'+ci]].plot(figsize=figsize)        
        plt.savefig(join(outdir,'sensor_'+ci+'.png'))
        plt.close()

    for ci in sensors:
        plot_comparison(results, ci)
        plt.title(ci+' - 5min')
        plt.savefig(join(outdir,'sensor_'+ci+'_scatter_5min.png'))
        plt.close()

        print('Resampling to 15min and 1h for sensor', ci)
        res15min = resample(results,'15min')
        plot_comparison(res15min, ci)
        plt.title(ci+' - 15min')
        plt.savefig(join(outdir,'sensor_'+ci+'_scatter_15min.png'))
        plt.close()

        res1h = resample(results,'1h')
        plot_comparison(res1h, ci)
        plt.title(ci+' - 1h')
        plt.savefig(join(outdir,'sensor_'+ci+'_scatter_1h.png'))
        plt.close()

        res1D = resample_sum(results,'1D')
        plot_comparison(res1D, ci,  coloring = 'M', label = 'Global Radiation (MJ m$^{-2}$ d$^{-1}$)')
        plt.title(ci+' - 1D')
        plt.savefig(join(outdir,'sensor_'+ci+'_scatter_1D.png'))
        plt.close()

        val = (results[ci]-results['simulated_'+ci]).abs()/results['ghi']
        val.plot(figsize=figsize)
        plt.axhline(y=val.mean(), color='red', linestyle='--')
        plt.savefig(join(outdir,'sensor_'+ci+'_error.png'))
        plt.close()

    fig, axarr = plt.subplots(2, 4, figsize=(12, 5), sharey=True)

    days = ['2023/06/05','2023/07/05','2023/08/05','2023/10/05']

    for i, ci in enumerate(sensors):
        for j, day in enumerate(days):
            ax = axarr[i, j]
            oneday = meteo_select_dates(results, day)
            print(np.unique(oneday.index.date))
            assert len(np.unique(oneday.index.date)) == 1

            ax.plot(oneday.index, oneday[ci],
                    color='orange', linewidth=1.5, label='observed')
            ax.plot(oneday.index, oneday['simulated_' + ci],
                    color='black', linewidth=1.2, label='simulated')

            # titres colonnes (dates)
            if i == 0:
                ax.set_title(day.replace('/', '-'), fontsize=10)

            # labels lignes (c2, c3 à droite)
            if j == 3:
                ax.text(1.05, 0.5, ci, transform=ax.transAxes,
                        rotation=90, va='center')
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,6,12,18,24]))

            # alléger les axes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # labels globaux
    fig.text(0.5, 0.04, 'Time (h)', ha='right')
    fig.text(0.04, 0.5, r'Global radiation (W m$^{-2}$)', va='center', rotation='vertical')

    # légende globale (une seule)
    handles, labels = axarr[0,0].get_legend_handles_labels()
    fig.legend(handles, labels,
           loc='upper center',
           ncol=2,
           frameon=False)
    plt.tight_layout(rect=[0.05, 0.1, 1, 0.95]) 
    plt.savefig(join(outdir,'sensor_exampledays.png'))
    plt.close()





if __name__ == '__main__':
    # date(month,day,hour)
    setup_meteo()
    assert meteo is not None, "Meteo data not loaded. Please check the setup_meteo function."
    results = process_sensors(meteo, view=True)
    plot_result(results)

