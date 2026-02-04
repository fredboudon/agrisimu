
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


""" Ne considerez que du '17-May' au '31-Oct' """
def process_light(meteo=meteo, mindate = None, maxdate = None, sensorheight = 0.0,  view = True, outdir = 'result'):
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

    # an agrivoltaic scene (generate plot)
    agrisystem = agristructure()
    fieldsensors = sensorgeometry(sensorheight)
    scene = fieldsensors+agrisystem

    l = LightEstimator(scene)
    l.localize(name = 'Camargue', **localisation)

    results = []
    l.set_method(method=eTriangleProjection, primitive=eShapeBased, occludedOnly = set([sh.id for sh in fieldsensors]), occludingOnly = set([sh.id for sh in agrisystem]))
    nbdates = len(meteo.index)
    initt = time.time()
    firsti = -1

    l.always_precompute()
    l.load_precomputation()

    for i, (cdate, row) in enumerate(meteo.iterrows()):
        globalirr, diffuseirr, c2, c3 = row['ghi'],row['dhi'],row['c2'],row['c3']
        if (mindate is None or cdate >= mindate) and (maxdate is None or cdate < maxdate) and globalirr > 0:
            t = time.time()
            #l.add_sun(dates = [cdate], irradiance = globalirr)
            fname = join(outdir,'simulation_'+str(sensorheight).replace('.','_')+'_'+cdate.strftime('%Y-%m-%d-%H-%M'))
            if os.path.exists(fname+'.csv'):
                results.append((cdate,fname+'.csv'))
                print(cdate,':  already done, skip ...')
                if firsti == i-1:
                    firsti = i
                continue
            l.clear_lights()
            l.add_sun_sky(dates = [cdate], ghi = globalirr, dhi = diffuseirr)
            print("Configure date '",cdate,"' with ghi:", globalirr, 'and dhi:', diffuseirr,'in', time.time()-t,'sec')

            result = l()

            t_res = time.time()
            result = format_values(result)
            result['TrIrradiance'] = result['irradiance']/globalirr

            if outdir:
                result.to_csv(fname+'.csv',sep='\t')
                results.append((cdate,fname+'.csv'))
                if False :
                    toimage(matrix_values(result, property='irradiance'), fname=fname+'_irradiancemap.png')
                    if len(l.lights) > 0:
                        plt.ion()
                        l.plot_sky()
                        plt.savefig(fname+'_skymap.png')
                        plt.close()
                    #l.scenerepr()[0].save(fname+'.bgeom')
            else:
                results.append((cdate,result))
            print('Generate output in', time.time()-t_res,'sec')
            simutime = time.time()-t
            #print(firsti,i,nbdates-i-1,time.time()-initt)
            estimate = (time.time()-initt)*(nbdates-i-1)/(i-firsti)
            if estimate > 3600:
                estimate_str = str(int(estimate/3600))+'h'+str(int((estimate%3600)/60))+'m'
            elif estimate > 60:
                estimate_str = str(int(estimate/60))+'m'+str(int(estimate%60))+'s'
            else:
                estimate_str = str(int(estimate))+'s'

            l.dump_precomputation()
            print('  simulation time:', simutime,'s - estimate',estimate_str)
   
    return results

if __name__ == '__main__':
    # date(month,day,hour)
    results = process_light(outdir='result', view=True)
    print(results)

