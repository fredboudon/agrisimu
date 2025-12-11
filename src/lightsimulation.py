
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
def process_light(mindate = date(5,1,0), maxdate = date(11, 1,0), view = True, outdir = None):
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

    # read the meteo
    meteo = read_meteo()

    # an agrivoltaic scene (generate plot)
    height = 0.0
    scene = generate_plots()+sensorgeometry(height)
    #scene = ground(height)
    
    initdate = date(1,1,0)

    sensordict = { 'c2' : Vector3(11,8,2), 'c3' : Vector3(27,4.5,2) }
    l = LightEstimator(scene)
    l.localize(name = 'Camargue', **localisation)
    l.add_sky(1)
    l.precompute_lights(type='SKY')
    for name, pos in sensordict.items():
        l.add_sensor(name, pos)
        l.sensors[name].compute()
        if view:
            plt.ion()
            l.sensors[name].view()
            plt.savefig('sensor_'+name+'_skymap.png')
            plt.close()

    results = []
    for index, row in meteo.iterrows():
        time, globalirr, diffuseirr, temperature = row
        cdate = initdate+datetime.timedelta(seconds=time)
        if cdate >= mindate and cdate < maxdate and globalirr > 0:
            print(time, cdate, globalirr, diffuseirr, temperature)
            l.add_sun_sky(dates = [cdate], ghi = globalirr, dhi = diffuseirr)
            #print('Lights', l.lights)
            #result = l( primitive=eShapeBased, screenresolution=RESOLUTION)
            result = l(method=eTriangleProjection, primitive=eShapeBased)
            print(result)
            result = format_values(result)
            result['TrIrradiance'] = result['irradiance']/globalirr
            result_sensor = l.estimate_sensors()

            if outdir:
                fname = join(outdir,'simulation_'+str(height).replace('.','_')+'_'+cdate.strftime('%Y-%m-%d-%H-%M'))
                result.to_csv(fname+'.csv',sep='\t')
                result_sensor.to_csv(fname+'_sensors.csv',sep='\t')
                toimage(matrix_values(result, property='irradiance'), fname=fname+'_irradiancemap.png')
                #toimage(matrix_values(result, property='TrIrradiance'), fname=fname+'_TrIrradiancemap.png')
                if len(l.lights) > 0:
                    plt.ion()
                    l.plot_sky()
                    plt.savefig(fname+'_skymap.png')
                    plt.close()
                if view:
                    l.scenerepr()[0].save(fname+'.bgeom')
            print(result)
            results.append((cdate,result))
    return results

if __name__ == '__main__':
    # date(month,day,hour)
    results = process_light(date(11,1,0), date(11,3,0), outdir='result', view=True)
    print(results)

