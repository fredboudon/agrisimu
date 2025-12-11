
import pandas
from openalea.plantgl.all import *

import os, sys

from importlib import reload
import generateplot; reload(generateplot)
from generateplot import *

import time, datetime, pytz
from os.path import join

import matplotlib.pyplot as plt


DEBUG = False
RESOLUTION = 0.1

localisation = {'latitude':43.734286, 'longitude':4.570565, 'altitude': 0, 'timezone': 'Europe/Paris'}
# north = -90

def read_meteo(data_file='weather.txt', localisation = localisation['timezone']):
    """ reader for mango meteo files """
    import pandas
    data = pandas.read_csv(data_file, delimiter = '\t',
                               usecols=['Time','Global','Diffuse','Temp'], dayfirst=True)

    data = data.rename(columns={'Time':'Time',
                                 'Global':'global_radiation',
                                 'Temp':'temperature_air'})
    # convert kW.m-2 to W.m-2
    #data['global_radiation'] *= 1000. 
    #index = pandas.DatetimeIndex(data['date']).tz_localize(localisation)
    #data = data.set_index(index)
    return data


def format_values(irradiances):
    irradiances = irradiances[irradiances.index >= IDDECAL]
    rowcol = []
    for shapeid, row  in irradiances.iterrows():
        rowcol.append(id2position(shapeid))
    cols, rows = zip(*rowcol)
    irradiances.insert(0,'row', rows)
    irradiances.insert(1,'column', cols)
    irradiances = irradiances.sort_values(['column','row'])
    irradiances = irradiances.reset_index(drop=True)

    return irradiances

def matrix_values(df, property='irradiance'):
    import numpy as np
    res = np.zeros((max(df['row'])+1,max(df['column'])+1))
    for index, row in df.iterrows():
        res[int(row['row']),int(row['column'])] = float(row[property])
    return res

def toimage(array, fname='out.png'):
    plt.imshow(array, cmap='jet',vmin = 0, origin='lower')
    plt.colorbar()
    plt.show(block=False)
    plt.savefig(fname)
    plt.close()

tz = pytz.timezone(localisation['timezone'])
def date(month, day, hour):
    return pandas.Timestamp(datetime.datetime(2023, month, day, hour, 0, 0), tz=tz)

""" Ne considerez que du '17-May' au '31-Oct' """
def process_light(mindate = date(5,1,0), maxdate = date(11, 1,0), meteofile = 'weather.txt', sensorheight = 0.0, view = True, outdir = 'result'):
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

    # read the meteo
    meteo = read_meteo(meteofile)

    # an agrivoltaic scene (generate plot)
    scene = generate_plots()+sensorgeometry(sensorheight)
    
    initdate = date(1,1,0)

    l = LightEstimator(scene)
    l.localize(name = 'Camargue', **localisation)

    print('Set diffuse irradiance map as precomputed...')
    l.add_sky(1)
    l.precompute_lights(type='SKY')

    results = []
    for index, row in meteo.iterrows():
        time, globalirr, diffuseirr, temperature = row
        cdate = initdate+datetime.timedelta(seconds=time)
        if cdate >= mindate and cdate < maxdate and globalirr > 0:
            print(time, cdate, globalirr, diffuseirr, temperature)
            l.clear_lights()
            l.add_sun_sky(dates = [cdate], ghi = globalirr, dhi = diffuseirr)

            fname = join(outdir,'simulation_'+str(sensorheight).replace('.','_')+'_'+cdate.strftime('%Y-%m-%d-%H-%M'))
            if os.path.exists(fname+'.csv'):
                results.append((cdate,fname+'.csv'))
                print('  already done, skip')
                continue
            result = l(method=eTriangleProjection, primitive=eShapeBased)
            result = format_values(result)
            result['TrIrradiance'] = result['irradiance']/globalirr

            if outdir:
                result.to_csv(fname+'.csv',sep='\t')
                results.append((cdate,fname+'.csv'))
                if view :
                    toimage(matrix_values(result, property='irradiance'), fname=fname+'_irradiancemap.png')
                    if len(l.lights) > 0:
                        plt.ion()
                        l.plot_sky()
                        plt.savefig(fname+'_skymap.png')
                        plt.close()
                    l.scenerepr()[0].save(fname+'.bgeom')
            else:
                results.append((cdate,result))
    return results

if __name__ == '__main__':
    results = process_light(date(5,1,0), date(5,2,0), outdir='result', view=True)
    print(results)

