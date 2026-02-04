
import pandas
import numpy as np
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


sensordict = { 'c2' : Vector3(29.5,4.5,2), 'c3' : Vector3(22,8,2) }
DEBUG = False
RESOLUTION = 0.1

# pilone en bas a gauche 1,5m en X et 1m en Y
# en carré                3           2
# Le suivant              3           8



def estimate_sensors(meteo, sensordict=sensordict, verbose=True):


    # an agrivoltaic scene (generate plot)
    height = 0.0
    scene = agristructure()
    
    l = LightEstimator(scene)
    l.localize(name = 'Camargue', **localisation)

    splitdate = '2023/6/21 12:0:0'
    meteo90cm = meteo[meteo.index <= splitdate]
    meteo2m = meteo[meteo.index > splitdate]

    results_date = []
    results_values = {} 
    t = time.time()

    for meteo, h in [(meteo90cm, 0.9), (meteo2m,2)]:    
        l.clear_sensors()
        sensorinfo = sensordict.copy()
        for v in sensorinfo.values():
            v.z = h
        print('Set sensor to positions:', sensordict)
        for sensorid, position in sensorinfo.items():
            if verbose:
                print('sensor', sensorid)
            l.add_sensor(sensorid, position)
            l.sensors[sensorid].compute()
        if verbose:
            print('Added', len(l.sensors), 'sensors')

        
        i = 0
        for cdate, row in meteo.iterrows():
            i += 1
            ghi, dhi = row['ghi'], row['dhi']
            if verbose and i %  100 == 0:  print('Processing', cdate, '...', i, ghi, dhi)
            l.clear_lights()
            l.add_light(f"sun_{cdate.strftime('%Y%m%d_%H%M')}", row['elevation'], row['azimuth'],  1, horizontal=True, date=cdate, type='SUN')
            result = l.estimate_sensors()
            results_date.append(cdate)
            for sensorid, irradiance in result.items():
                if sensorid not in results_values:
                    results_values[sensorid] = []
                results_values[sensorid].append(irradiance)
    if verbose:
        print('Done '+str(len(results_date))+' simulations in', time.time() - t, 'seconds')
    result ={'date_time': results_date}
    result.update(results_values)
    results = pandas.DataFrame(result)
    index = results['date_time']
    results.drop(columns=["date_time"], inplace=True)
    results = results.set_index(index)
    results = results.join(meteo[['elevation','azimuth','zenith','c2shaded','c3shaded']], how='inner')
    return results



def diff_map(a,b):
    return (a == 0) == b

def error_func(a,b):
    return np.sum(diff_map(a,b)/len(a))

def _launch_light_simulation(args):
    test_coord, sensordict, meteo = args
    allthesame = False
    if len(sensordict) > 1:
        sensorpos = list(sensordict.values())
        if sensorpos == [sensorpos[0] for pos in sensorpos]:
            allthesame = True
            _sensordict = sensordict
            sensordict = {'ci' : sensorpos[0]}
    results = estimate_sensors(sensordict=sensordict, meteo=meteo, verbose=False)
    # shaded simulation have 0 as result
    if allthesame:
        for sensorid in _sensordict.keys():
            results[sensorid] = results['ci']
        sensordict = _sensordict
    total_irradiance_error = {sensorid: error_func(results[sensorid], results[sensorid+'shaded']) for sensorid in sensordict.keys()}
    return test_coord, sensordict, total_irradiance_error

import multiprocessing
import itertools
def bruteforce_argmax(a):
    ri, rj = 0,0
    minv = a[0,0]
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i,j] > minv:
                minv = a[i,j]
                ri, rj = i,j
    return ri, rj

def optimize_sensor_position(sensordict=sensordict, meteo= clearsky1, step = 0.01, rangetotest = 1, shift = 0, multithreading = True, outdir='optimization_result'):
    best_positions = {}
    nbjobs = multiprocessing.cpu_count() * 5
    t = time.time()

    fullmap = [None for _ in range(len(sensordict))] == list(sensordict.values())

    if fullmap:
        if isinstance(rangetotest,tuple):
            print(shift,rangetotest[0],step)
            print(shift,rangetotest[1],step)
            xdecalvalues = np.arange(shift,rangetotest[0]+1e-3,step)
            ydecalvalues = np.arange(shift,rangetotest[1]+1e-3,step)
        else:
            xdecalvalues = np.arange(shift,rangetotest+1e-3,step)
            ydecalvalues = np.arange(shift,rangetotest+1e-3,step)
    else:
        if isinstance(rangetotest,tuple):
            xdecalvalues = np.arange(-rangetotest[0]/2+shift,(rangetotest[0]/2)+1e-3,step)
            ydecalvalues = np.arange(-rangetotest[1]/2+shift,(rangetotest[1]/2)+1e-3,step)
        else:
            xdecalvalues = np.arange(-rangetotest/2+shift,(rangetotest/2)+1e-3,step)
            ydecalvalues = np.arange(-rangetotest/2+shift,(rangetotest/2)+1e-3,step)

    positions_to_test = []
    allpositions = [[Vector3(0,0,0) for _ in ydecalvalues] for _ in xdecalvalues]
    for i, dx in enumerate(xdecalvalues):
        for j,dy in enumerate(ydecalvalues):
            positions_to_test.append((i,j))
            allpositions[i][j] = Vector3(dx,dy,0)

    def tosensorpos(i,j, sensorid):
        if sensordict[sensorid] is None:
            return allpositions[i][j]+Vector3(0,0,2)
        else:
            return sensordict[sensorid]+allpositions[i][j]

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    resultfnames = {sensorid : join(outdir,'irradiance_error_map_loc_'+sensorid+'_'+str(step)+'_'+str(rangetotest)+'_'+('fullmap' if fullmap else '_'.join(map(str,sensorpos)))) for sensorid,sensorpos in sensordict.items()}
    if all([os.path.exists(f+'.npy') for f in resultfnames.values()]):
        print('Load optimization from',resultfnames)
        errormap = { sensorid : np.load(fname+'.npy') for sensorid, fname in resultfnames.items() }
        firstmap = list(errormap.values())[0]
        lastposition = (firstmap.shape[0]-1,firstmap.shape[1]) 
        for sensorid in sensordict.keys():
            for dx, dy in itertools.product(range(0,errormap[sensorid].shape[0]), range(0,errormap[sensorid].shape[1])):
                    if errormap[sensorid][dx,dy] < 0 :
                        if lastposition[0] > dx or (lastposition[0] == dx and lastposition[1] > dy):
                            lastposition = (dx,dy)
                            print('Found incomplete optimization at position', lastposition, errormap[sensorid][dx,dy])
                        break
        startindex = lastposition[0]*len(ydecalvalues)+lastposition[1]
        print('Resuming optimization from position index', startindex, 'out of', len(positions_to_test))
        #positions_to_test = positions_to_test[startindex:]
    else:
        print('No already computed data')
        errormap = { sensorid : np.full((len(xdecalvalues),len(ydecalvalues)),-1., dtype=float) for sensorid in sensordict.keys() }
        startindex = 0

    print('Optimize sensor', list(sensordict.keys()), 'over', len(meteo), 'meteo samples and', len(positions_to_test), 'positions...')
    if not fullmap:
        print('Start from', sensordict)
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
                allvalues = pool.map(_launch_light_simulation, [((i,j), dict([(sensorid, tosensorpos(i,j, sensorid)) for sensorid in sensordict.keys()]), meteo) for i,j in batch_positions])
        else:
            allvalues = [_launch_light_simulation(((i,j), dict([(sensorid, tosensorpos(i,j, sensorid)) for sensorid in sensordict.keys()]), meteo)) for i,j in batch_positions]
        for test_coord, test_position, total_irradiance_error in allvalues:
            for sensorid in sensordict.keys():
                assert total_irradiance_error[sensorid] >= 0
                errormap[sensorid][test_coord[0], test_coord[1]] = total_irradiance_error[sensorid]
        for sensorid, fname in resultfnames.items():
            print('Intermediary save of irradiance error map for sensor', sensorid, 'from', i, 'to', min(i+nbjobs, len(positions_to_test)))
            np.save(fname+'.npy', errormap[sensorid])
            # toimage(errormap[sensorid], fname=fname+'.png', vmin = errormap[sensorid].min())

    best_irradiance_errors = {sensorid: 1e100 for sensorid in sensordict.keys()}
    best_positions = {sensorid: None for sensorid in sensordict.keys()}
    best_coords = {sensorid: None for sensorid in sensordict.keys()}
    for sensorid in sensordict.keys():
        ix, iy = np.unravel_index(np.argmax(errormap[sensorid]),errormap[sensorid].shape)
        best_irradiance_errors[sensorid] = errormap[sensorid][ix,iy]
        best_positions[sensorid] = tosensorpos(ix, iy, sensorid)
        best_coords[sensorid] = (ix,iy)

    best_sensor_result = estimate_sensors(meteo, best_positions, verbose=False)
    for sensorid, fname in resultfnames.items():
        print('Best position for sensor', sensorid, 'is', best_positions[sensorid], 'with irradiance error=', best_irradiance_errors[sensorid])
        np.save(fname+'.npy', errormap[sensorid])
        print("Save",repr(fname+'.png'))
        print(tosensorpos(0,0,sensorid).x,tosensorpos(-1,0,sensorid).x+1e-3)        
        print(tosensorpos(0,0,sensorid).y,tosensorpos(0,-1,sensorid).y+1e-3)
        toimage(errormap[sensorid].T, fname=fname+'.png', vmin=errormap[sensorid].min(), 
                yticklabels=list(map(lambda x : str(round(x,2)),np.arange(tosensorpos(0,0,sensorid).x,tosensorpos(-1,0,sensorid).x+1e-3,(max(xdecalvalues) - min(xdecalvalues))/10))),
                xticklabels=list(map(lambda x : str(round(x,2)),np.arange(tosensorpos(0,0,sensorid).y,tosensorpos(0,-1,sensorid).y+1e-3,(max(ydecalvalues) - min(ydecalvalues))/10))),
                markpixel=list(reversed(best_coords[sensorid])))

        plot_meteo(best_sensor_result, best_sensor_result[sensorid], polar=True, cmap='binary_r', marker='.', blocking=False)
        fname_view = join(outdir,'view_'+sensorid+'_'+'_'.join(map(lambda x : str(round(x,3)),best_positions[sensorid]))) 
        plt.savefig(fname_view+'.png')
        plt.close()
        print("Save view :", repr(fname_view))
        plot_meteo(best_sensor_result, (best_sensor_result[sensorid] ==0)*2+ best_sensor_result[sensorid+'shaded'], polar=True, cmap='jet', marker='.', blocking=False)
        plt.savefig(fname_view+'_diff.png')
        plt.close()
    print('Optimization done in', time.time() - t, 'seconds.')
    print('Initial positions are:', sensordict)
    print('Best positions are:', best_positions)

    return best_positions

def optimize_sensor_position_fullmap(sensordict=sensordict, meteo= clearsky1, step = 0.5,  multithreading = True, outdir='optimization_result'):
    return optimize_sensor_position(sensordict={sid : None for sid in sensordict}, meteo= clearsky1, step = step, 
                                    rangetotest = (MAPLENGTH, MAPWIDTH), shift = step/2, multithreading = multithreading, outdir=outdir)


if __name__ == '__main__':
    #sensordict = optimize_sensor_position_fullmap()
    optimize_sensor_position(sensordict)
    #sensors = estimate_sensors(clearsky)
    #plot_meteo(sensors, sensors['c2'], polar=True, cmap='binary_r', marker='.')
    pass

