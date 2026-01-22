
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

def read_meteo(data_file='2023_PAR_c1.txt', localisation = localisation['timezone']):
    """ reader for mango meteo files """
    import pandas
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
    sky_irr = sky_irradiance(dates=ppfd.index, ghi=global_radiation['Rg'], ppfd=ppfd['PAR'])
    return sky_irr



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
    meteo = get_meteo('c1')

    # an agrivoltaic scene (generate plot)
    height = 0.0
    scene = generate_plots()
    #scene = ground(height)
    

    sensordict = { 'c2' : Vector3(57*0.5,9*0.5,2), 'c3' : Vector3(43*0.5,15*0.5,2) }
    l = LightEstimator(scene)
    l.localize(name = 'Camargue', **localisation)

    for sensorid, position in sensordict.items():
        print('sensor', sensorid)
        l.add_sensor(sensorid, position)
        l.sensors[sensorid].compute()
        if view and outdir:
                plt.ion()
                l.sensors[sensorid].view()
                plt.savefig(join(outdir,'sensor_'+sensorid+'_skymap.png'))
                plt.close()

        #l.sensors[sensorid].view()

    print('Added', len(l.sensors), 'sensors')

    results_date = []
    results_values = {'ghi': [], 'dhi': []  }
    i = 0
    for cdate, row in meteo.iterrows():
        i += 1
        ghi, dhi = row['ghi'], row['dhi']
        if (mindate is None or cdate >= mindate) and (maxdate is None or cdate < maxdate) and ghi > 0:
            if i %  100 == 0:  print('Processing', cdate, '...', i, ghi, dhi)
            l.clear_lights()
            l.add_astk_sun_sky(dates = [cdate], ghi = ghi, dhi = dhi)
            result = l.estimate_sensors()
            results_date.append(cdate)
            for sensorid, irradiance in result.items():
                if sensorid not in results_values:
                    results_values[sensorid] = []
                results_values[sensorid].append(irradiance)
                results_values['ghi'].append(ghi)
                results_values['dhi'].append(dhi)

    result ={'date_time': results_date}
    result.update(results_values)
    results = pandas.DataFrame(result)
    index = results['date_time']
    results.drop(columns=["date_time"], inplace=True)
    results = results.set_index(index)
    pos2str = lambda pos: str(pos.x)+'_'+str(pos.y)+'_'+str(pos.z)
    if outdir:
        results.to_csv(join(outdir,'sensors_irradiance_'+pos2str(sensordict['c2'])+'_'+pos2str(sensordict['c3'])+'.csv'),sep='\t')
    return results

if __name__ == '__main__':
    # date(month,day,hour)
    #results = process_light(date(5,1,0), date(5,2,0), outdir='result', view=True)
    #results = process_light( outdir='result', view=True)
    results = process_light(None, None, outdir='result', view=True)
    print(results)
    #print(meteo('c2'))

