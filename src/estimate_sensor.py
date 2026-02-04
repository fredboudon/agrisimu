
import pandas
from openalea.plantgl.all import *

import os, sys

from importlib import reload
import generateplot; reload(generateplot)
from generateplot import *

import data_util; reload(data_util)
from data_util import *

import time, datetime, pytz
from os.path import join

import matplotlib.pyplot as plt


DEBUG = False
RESOLUTION = 0.01  # in meters

sensordict = {'c2': Vector3(35.43,0.19,2), 'c3': Vector3(34.64,2.89,2)}

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
        os.mkdir(outdir)

    if not mindate is None and maxdate is None:
         maxdate = mindate + datetime.timedelta(1)

    # an agrivoltaic scene (generate plot)
    scene = agristructure()


    l = LightEstimator(scene)
    l.localize(name = 'Camargue', **localisation)

    print('Set diffuse irradiance map as precomputed...')

    for name, pos in sensordict.items():
            l.add_sensor(name, pos)
            l.sensors[name].compute()
            if view:
                plt.ion()
                l.sensors[name].view()
                if outdir:
                    plt.savefig(join(outdir,'sensor_'+name+'_skymap.png'))
                plt.close()

    results = {sensorid : [] for sensorid in sensordict.keys()}
    resultdate = []
    results['date_time'] = resultdate

    t = time.time()
    for cdate, row in meteo.iterrows():
        globalirr, diffuseirr, c2, c3 = row['ghi'],row['dhi'],row['c2'],row['c3']
        if (mindate is None or cdate >= mindate) and (maxdate is None or cdate < maxdate) and globalirr > 0:
            print(cdate, globalirr, diffuseirr)
            l.clear_lights()
            l.add_sun_sky(dates = [cdate], ghi = globalirr, dhi = diffuseirr)
            #fname = join(outdir,'simulation_'+str(sensorheight).replace('.','_')+'_'+cdate.strftime('%Y-%m-%d-%H-%M'))
            result_sensor = l.estimate_sensors()
            resultdate.append(cdate)
            for sid, value in result_sensor.items():
                 results[sid].append(value)

    print('  simulation time:', time.time()-t)

    results = pandas.DataFrame(results)
    index = pandas.DatetimeIndex(results['date_time'])
    results.drop(columns=["date_time"], inplace=True)
    results = results.set_index(index)
    results.rename(columns={sensorid : 'simulated_'+sensorid for sensorid in sensordict.keys()}, inplace=True)
    results = results.join(meteo[np.isin(meteo.index, results.index)])
    
    if outdir:
        fname = 'sensor_simulation'
        if not mindate is None:
             fname+='_'+str(mindate)
        if not maxdate is None:
             fname+='_'+str(maxdate)        
        results.to_csv(join(outdir,fname+'.csv'),sep='\t')
        
    return results

def plot_result(results):
     oneday = meteo_select_dates(results,'2023/06/01')
     for ci in ['c2','c3']:
     results[[ci,'simulated_'+ci]].plot()
     plt.show()
     (results[ci]-results['simulated_'+ci]).plot()
     plt.show()
     oneday[[ci,'simulated_'+ci]].plot()
     plt.show()


if __name__ == '__main__':
    # date(month,day,hour)
    results = process_sensors(meteo, view=True)
    print(results)

