
import pandas
from openalea.plantgl.all import *

import os, sys

from importlib import reload
import generateplot; reload(generateplot)
from generateplot import *

import data_util; reload(data_util)
from data_util import *

from localization import *

import time
from os.path import join

import matplotlib.pyplot as plt


DEBUG = False
RESOLUTION = 0.01  # in meters


def _process_light(*args):
    return process_light(*args[0], multithreaded = False)

def process_light(meteo=None, mindate = None, maxdate = None, outdir = 'result', jobname = None, sensorheight = 0, multithreaded = True):
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
    if jobname:
        print('Start job', jobname)
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)
    
    if os.path.exists(join(outdir,'global_irradiances.pkl')):
        print('Job', jobname, ': results already exist, skip ...')
        return []

    # an agrivoltaic scene (generate plot)
    agrisystem = agristructure()
    fieldsensors = sensorgeometry(sensorheight)
    scene = fieldsensors+agrisystem
    precomputationpath = '.pglcache'+str(sensorheight).replace('.','_')

    l = LightEstimator(scene)
    l.localize(name = 'Camargue', **localisation)

    results = []
    l.set_method(method=eTriangleProjection, primitive=eShapeBased, occludedOnly = set([sh.id for sh in fieldsensors]), occludingOnly = set([sh.id for sh in agrisystem]), multithreaded = multithreaded)
    nbdates = len(meteo.index)
    initt = time.time()
    firsti = -1

    l.always_precompute()
    l.load_precomputation(precomputationpath)

    prefixmessage = '**** ' +('job '+str(jobname).zfill(2) if not jobname is None else 'main')+ ' -'

    for i, (cdate, row) in enumerate(meteo.iterrows()):
        globalirr, diffuseirr = row['ghi'],row['dhi']
        if (mindate is None or cdate >= mindate) and (maxdate is None or cdate < maxdate) and globalirr > 0:
            t = time.time()
            fname = join(outdir,'simulation_'+str(sensorheight).replace('.','_')+'_'+cdate.strftime('%Y-%m-%d-%H-%M'))
            if os.path.exists(fname+'.csv'):
                results.append((cdate,fname+'.csv'))
                print(prefixmessage, "Process date",repr(str(cdate)),"with ghi: %.2f and dhi: %.2f" % (globalirr, diffuseirr),':  already done, skip ...')
                if firsti == i-1:
                    firsti = i
                continue
            l.clear_lights()
            l.add_light(f"sun_{cdate.strftime('%Y%m%d_%H%M')}", row['elevation'], row['azimuth'],  globalirr-diffuseirr, horizontal=True, date=cdate, type='SUN')
            l.add_sky(irradiance = diffuseirr)
            print(prefixmessage, "Process date",repr(str(cdate)), "with ghi: %.2f and dhi: %.2f" % (globalirr, diffuseirr), '...')

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
            #print('Generate output in', time.time()-t_res,'sec')
            simutime = time.time()-t
            #print(firsti,i,nbdates-i-1,time.time()-initt)
            estimate = (time.time()-initt)*(nbdates-i-1)/(i-firsti)
            if estimate > 3600:
                estimate_str = str(int(estimate/3600))+'h'+str(int((estimate%3600)/60))+'m'
            elif estimate > 60:
                estimate_str = str(int(estimate/60))+'m'+str(int(estimate%60))+'s'
            else:
                estimate_str = str(int(estimate))+'s'

            l.dump_precomputation(precomputationpath)
            print(prefixmessage, "Process date",repr(str(cdate)),':  simulation time :', simutime,'s - estimate',estimate_str)
    
    if outdir:
        print(prefixmessage, "All results saved in directory", outdir, "from", repr(str(mindate)), "to", repr(str(maxdate)))
        if jobname is None:
            print(prefixmessage, "Total simulation time : ", time.time()-initt,'s')
            print(prefixmessage, "Compress irradiance results ...")
            compress_irradiances(outdir)
   
    return results

def mt_process_light(meteo=None, mindate = None, maxdate = None, nbjobs=None,**kwargs):
    import os
    outdir = kwargs.get('outdir','result')
    if os.path.exists(join(outdir,'global_irradiances.pkl')) or os.path.exists(join(outdir,'global_irradiances.pkl.zip')):
        print('results already exist, skip ...')
        return load_irradiances(outdir)
    if not mindate is None:
        meteo = meteo[meteo.index >= mindate]
    if not maxdate is None:
        meteo = meteo[meteo.index < maxdate]
    nbitems = len(meteo.index)
    nbjobs  = os.cpu_count() if nbjobs is None else nbjobs
    itemsperjob = (nbitems+nbjobs-1)//nbjobs
    args = []
    for i in range(nbjobs):
        submeteo = meteo.iloc[i*itemsperjob:(i+1)*itemsperjob]
        if len(submeteo) > 0:
            args.append((submeteo, mindate, maxdate, kwargs.get('outdir','result'), i ))
    print('Run', nbjobs, 'parallel jobs with', itemsperjob, 'items each (last one with', len(submeteo), 'items)')
    if nbjobs == 1:
        results = [ process_light(*arg) for arg in args ]   
    else:
        from multiprocessing import Pool
        results = Pool().map(_process_light, args)
    compress_irradiances(kwargs.get('outdir','result'))
    return results

if __name__ == '__main__':
    from meteo import *
    setup_meteo()
    # date(month,day,hour)
    #results = mt_process_light(outdir='result/weather2023', meteo=meteo)
    #results = mt_process_light(outdir='result/clear_sky', meteo=generate_meteo())
    #results = mt_process_light(outdir='result/cloudy', meteo=generate_meteo(attenuation=0.3))
    results = mt_process_light(outdir='result/intermediate_sky', meteo=generate_meteo(attenuation=0.5))
    #print(results)

