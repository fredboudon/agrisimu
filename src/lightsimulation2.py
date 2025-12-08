
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
    rows = {}
    cols = {}
    for id, irr  in irradiances.items():
        rows[id], cols[id] = id2position(id)
    irradiances = pandas.DataFrame({'row':rows, 'column':cols, 'irradiance':irradiances})
    irradiances = irradiances.sort_values(['column','row'])

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
def process_light(mindate = date(5,1,0), maxdate = date(11, 1,0), view = True, outdir = None):
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

    # read the meteo
    meteo = read_meteo()

    # an agrivoltaic scene (generate plot)
    height = 0.0
    scene = generate_plots()
    #scene = ground(height)
    
    initdate = date(1,1,0)

    l = LightEstimator(scene)
    l.localize(name = 'Camargue', **localisation)
    for sensorid, position in sensorpositions(height):
        print('sensor', sensorid)
        l.add_sensor(sensorid, position)
        l.sensors[sensorid].compute()
        #l.sensors[sensorid].view()
    print('Added', len(l.sensors), 'sensors')

    results = []
    for index, row in meteo.iterrows():
        time, globalirr, diffuseirr, temperature = row
        cdate = initdate+datetime.timedelta(seconds=time)
        if cdate >= mindate and cdate < maxdate and globalirr > 0:
            l.clear_lights()
            print(time, cdate, globalirr, diffuseirr, temperature)
            l.add_astk_sun_sky(dates = [cdate], ghi = globalirr, dhi = diffuseirr)
            result = l.estimate_sensors()
            result = format_values(result)
            result['TrIrradiance'] = result['irradiance']/globalirr

            if outdir:
                fname = join(outdir,'sensors_'+str(height).replace('.','_')+'_'+cdate.strftime('%Y-%m-%d-%H-%M'))
                result.to_csv(fname+'.csv',sep='\t')
                toimage(matrix_values(result, property='irradiance'), fname=fname+'_irradiancemap.png')
                #toimage(matrix_values(result, property='TrIrradiance'), fname=fname+'_TrIrradiancemap.png')
                if len(l.lights) > 0:
                    plt.ion()
                    l.plot_sky()
                    plt.savefig(fname+'_skymap.png')
                    plt.close()
                if view:
                    (l.sensors_repr(size=0.5)[0]+l.scene).save(fname+'.bgeom')
                    #l.plot(sensorsize=0.5)
            results.append((cdate,result))
    return results

if __name__ == '__main__':
    # date(month,day,hour)
    results = process_light(date(5,1,0), date(5,2,0), outdir='result', view=True)
    results = process_light( outdir='result', view=True)
    #results = process_light(date(5,1,12), date(5,1,13), outdir='result', view=True)
    print(results)

