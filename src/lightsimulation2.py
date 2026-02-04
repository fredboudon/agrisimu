
import pandas
import numpy as np
from openalea.plantgl.all import *

import os, sys

from importlib import reload
import generateplot; reload(generateplot)
from generateplot import *

import time, datetime, pytz
from os.path import join

import matplotlib.pyplot as plt

#Best positions are: {'c2': Vector3(28,5.4,2), 'c3': Vector3(20.5,7.9,2)}
#Best positions are: {'c2': Vector3(27.9,5.4,2), 'c3': Vector3(20.5,8.4,2)}
DEBUG = False
RESOLUTION = 0.1

# pilone en bas a gauche 1,5m en X et 1m en Y
# en carré                3           2
# Le suivant              3           8

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
    sky_irr = sky_irradiance(dates=ppfd.index, ghi=global_radiation['Rg'], ppfd=ppfd['PAR'], latitude=localisation['latitude'], longitude=localisation['longitude'], altitude=localisation['altitude'])
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

sensordict = { 'c2' : Vector3(57*0.5,9*0.5,2), 'c3' : Vector3(43*0.5,15*0.5,2) }
# read the meteo
meteo = get_meteo('c1')
ref_data = { 'c2': read_meteo('2023_globalradiations_c2.txt')['Rg'], 'c3': read_meteo('2023_globalradiations_c3.txt')['Rg'] }

""" Ne considerez que du '17-May' au '31-Oct' """
def process_light(mindate = date(5,1,0), maxdate = date(11, 1,0), sensordict=sensordict, meteo=meteo, view = True, outdir = None, verbose = True):
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)


    # an agrivoltaic scene (generate plot)
    height = 0.0
    scene = agristructure()
    
    l = LightEstimator(scene)
    l.localize(name = 'Camargue', **localisation)

    print('Set sensor to positions:', sensordict)
    for sensorid, position in sensordict.items():
        if verbose:
            print('sensor', sensorid)
        l.add_sensor(sensorid, position)
        l.sensors[sensorid].compute()
        if view and outdir:
                plt.ion()
                l.sensors[sensorid].view()
                plt.savefig(join(outdir,'sensor_'+sensorid+'_skymap.png'))
                plt.close()
    if verbose:
        print('Added', len(l.sensors), 'sensors')

    results_date = []
    results_values = {} # {'ghi': [], 'dhi': []  }
    i = 0
    t = time.time()
    for cdate, row in meteo.iterrows():
        i += 1
        ghi, dhi = row['ghi'], row['dhi']
        if (mindate is None or cdate >= mindate) and (maxdate is None or cdate < maxdate) and ghi > 0:
            if verbose and i %  100 == 0:  print('Processing', cdate, '...', i, ghi, dhi)
            l.clear_lights()
            l.add_sun_sky(dates = [cdate], ghi = ghi, dhi = dhi)
            result = l.estimate_sensors()
            results_date.append(cdate)
            for sensorid, irradiance in result.items():
                if sensorid not in results_values:
                    results_values[sensorid] = []
                results_values[sensorid].append(irradiance)
            #results_values['ghi'].append(ghi)
            #results_values['dhi'].append(dhi)
    if verbose:
        print('Done '+str(len(results_date))+' simulations in', time.time() - t, 'seconds')
    result ={'date_time': results_date}
    result.update(results_values)
    results = pandas.DataFrame(result)
    index = results['date_time']
    results.drop(columns=["date_time"], inplace=True)
    results = results.set_index(index)
    if outdir:
        pos2str = lambda pos: str(pos.x)+'_'+str(pos.y)+'_'+str(pos.z)
        results.to_csv(join(outdir,'sensors_irradiance_'+str(sensordict['c2'])+'_'+str(sensordict['c3'])+'.csv'),sep='\t')
    #results['ref_c2'] = refc2
    #results['ref_c3'] = refc3
    return results



def _launch_light_simulation(args):
    sensorid, test_coord, test_position, meteo = args
    test_sensordict = dict(zip(sensorid,[test_position for _ in sensorid]))
    results = process_light(sensordict=test_sensordict, meteo=meteo, view=False, verbose=False)
    total_irradiance_error = {sensorid: (results[sensorid]-ref_data[sensorid]).abs().sum()/len(results) for sensorid in sensorid}
    return test_coord, test_position, total_irradiance_error

import multiprocessing
import itertools

def optimize_sensor_position(sensordict=sensordict, multithreading = True, outdir='optimization_result'):
    best_positions = {}
    dates = np.unique(meteo.index.date)
    nbdates =  50
    step = 0.05
    nbjobs = multiprocessing.cpu_count() * 5
    selected_dates = np.random.choice(dates, size=nbdates, replace=False)
    mask = np.isin(meteo.index.date, selected_dates)
    selected = meteo[mask]
    assert len(np.unique(selected.index.date)) == nbdates
    t = time.time()
    positions_to_test = []
    for i, dx in enumerate(np.arange(step,MAPLENGTH,step)):
        for j,dy in enumerate(np.arange(step,MAPWIDTH,step)):
            positions_to_test.append(((i,j), Vector3(dx,dy,2)))

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    resultfnames = {sensorid : join(outdir,'irradiance_error_map_'+sensorid+'_'+str(step)) for sensorid in sensordict.keys()}
    if all([os.path.exists(f+'.npy') for f in resultfnames.values()]):
        errormap = { sensorid : np.load(fname+'.npy') for sensorid, fname in resultfnames.items() }
        lastposition = (0,0) 
        for sensorid in sensordict.keys():
            print(errormap[sensorid][0,0:100])
        for sensorid in sensordict.keys():
            for dx, dy in itertools.product(range(0,errormap[sensorid].shape[0]), range(0,errormap[sensorid].shape[1])):
                    if errormap[sensorid][dx,dy] < 0 :
                        if lastposition[0] < dx or (lastposition[0] == dx and lastposition[1] < dy):
                            lastposition = (dx,dy)
                            print('Found incomplete optimization at position', lastposition, errormap[sensorid][dx,dy])
                        break
        startindex = lastposition[0]*int(MAPWIDTH/step)+lastposition[1]
        print('Resuming optimization from position index', startindex, 'out of', len(positions_to_test))
        #positions_to_test = positions_to_test[startindex:]
    else:
        errormap = { sensorid : np.full((int(MAPLENGTH/step)-1,int(MAPWIDTH/step)-1),-1., dtype=float) for sensorid in sensordict.keys() }
        startindex = 0

    print('Optimize sensor', list(sensordict.keys()), 'over', len(selected), 'meteo samples and', len(positions_to_test), 'positions...')
    t = time.time()
    for i in range(startindex, len(positions_to_test), nbjobs):
        batchtime = time.time()-t
        t = time.time()
        estimate = int(batchtime*(len(positions_to_test)-i)/nbjobs)
        if estimate > 3600:
            estimate_str = str(int(estimate/3600))+'h'+str(int((estimate%3600)/60))+'m'
        elif estimate > 60:
            estimate_str = str(int(estimate/60))+'m'+str(int(estimate%60))+'s'
        else:
            estimate_str = str(int(estimate))+'s'
        print('******['+str(int(100*i/len(positions_to_test)))+'%] ('+str(int(batchtime))+'sec - est. '+estimate_str+')*********  processing positions', i, 'to', min(i+nbjobs, len(positions_to_test)), 'from', len(positions_to_test))
        batch_positions = positions_to_test[i : min(i+nbjobs, len(positions_to_test))]
        if multithreading:
            with multiprocessing.Pool() as pool:
                allvalues = pool.map(_launch_light_simulation, [(list(sensordict.keys()), test_position[0], test_position[1], selected) for test_position in batch_positions])
        else:
            allvalues = [_launch_light_simulation((list(sensordict.keys()), test_position[0], test_position[1], selected)) for test_position in batch_positions]
        for test_coord, test_position, total_irradiance_error in allvalues:
            for sensorid in sensordict.keys():
                assert total_irradiance_error[sensorid] >= 0
                errormap[sensorid][test_coord[0], test_coord[1]] = total_irradiance_error[sensorid]
        for sensorid, fname in resultfnames.items():
            print('Intermediary save of irradiance error map for sensor', sensorid, 'from', i, 'to', min(i+nbjobs, len(positions_to_test)))
            np.save(fname+'.npy', errormap[sensorid])
            toimage(errormap[sensorid], fname=fname+'.png')

    best_irradiance_errors = {sensorid: 1e100 for sensorid in sensordict.keys()}
    best_positions = {sensorid: None for sensorid in sensordict.keys()}
    for sensorid in sensordict.keys():
        for dx in range(errormap[sensorid].shape[0]):
            for dy in range(errormap[sensorid].shape[1]):
                if errormap[sensorid][dx,dy] < best_irradiance_errors[sensorid]:
                    best_irradiance_errors[sensorid] = errormap[sensorid][dx,dy]
                    best_positions[sensorid] = Vector3((dx+1)*step, (dy+1)*step, 2)

    for sensorid, fname in resultfnames.items():
        print('Best position for sensor', sensorid, 'is', best_positions[sensorid], 'with irradiance error=', best_irradiance_errors[sensorid])
        np.save(fname+'.npy', errormap[sensorid])
        toimage(errormap[sensorid], fname=fname+'.png')
    print('Optimization done in', time.time() - t, 'seconds.')
    print('Best positions are:', best_positions)
    return best_positions

if __name__ == '__main__':
    # date(month,day,hour)
    #results = process_light(date(5,1,0), date(5,2,0), outdir='result', view=True)
    #results = process_light( outdir='result', view=True)
    #results = process_light(None, date(5,20,0), outdir='result', view=True)
    #print(results)
    #results[['ghi','ref_c2','c2']].plot()
    #pandas.DataFrame({'c2_error': results['c2']/results['ref_c2'], 'c3_error': results['c3']/results['ref_c3']}).plot()
    #print(meteo('c2'))
    bestpositions = optimize_sensor_position()
    #results = process_light(date(5,19,0), date(5,20,0), outdir='result', view=True)

