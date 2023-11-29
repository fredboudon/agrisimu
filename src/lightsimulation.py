
import pandas
from openalea.plantgl.all import *
from alinea.astk.sun_and_sky import sun_sky_sources, sun_sources
import os, sys, datetime

from importlib import reload
import generateplot; reload(generateplot)
from generateplot import *
import time

DEBUG = False

localisation={'latitude':43.734286, 'longitude':4.570565, 'timezone': 'Europe/Paris'}

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



def toCaribuScene(scene, opt_prop) :
    from alinea.caribu.CaribuScene import CaribuScene
    t = time.time()
    print ('Convert scene for caribu')
    nscene = Scene()
    cs = CaribuScene(scene, opt=opt_prop, scene_unit='m', debug = DEBUG)
    print('done in', time.time() - t)
    return cs


def caribu(scene, sun = None, sky = None, view = False, debug = False):
    from alinea.caribu.light import light_sources
    print('start caribu...')
    t = time.time()
    print('Create light source', end=' ')
    light = []
    if not sun is None:
        light += light_sources(*sun) 
    if not sky is None:
        light += light_sources(*sky)
    print('... ',len(light),' sources.')
    scene.setLight(light)
    print('Run caribu')
    #raw, agg = scene.run(direct=False, infinite = True, split_face = True, d_sphere = D_SPHERE)
    raw, agg = scene.run(direct=False, infinite = False, split_face = False)
    print('made in', time.time() - t)
    if view : 
        scene.plot(raw['PAR']['Ei'])
    return raw, agg


def normalize_energy(lights):
    el,az,ei = lights
    sumei = ei.sum()
    lights = el, az, ei / sumei
    return lights, sumei

import multiprocessing
import glob
import pickle
from random import randint
from os.path import join


def generate_dataframe_data(aggregatedResults, date, name, ei, gus, outdir):
    aggRc = filter_keys(aggregatedResults['Rc']['Ei'], gus)
    lres = dict()
    lres['Entity'] = ['incident']+list(aggRc)
    for wavelength in ['Rc','Rs','PAR']:
        for result in ['Ei','Ei_sup','Ei_inf','Eabs']:
            res = filter_res(aggregatedResults[wavelength][result], gus)
            if sum(np.array(res) < -1e-5) > 0:
                errormsg = str(datetime.datetime.now())+' : Error with '+name+' on '+wavelength+'-'+result+' : Negative values found.'
                print(errormsg)
                open(os.path.join(outdir,'error.log'),'a').write(errormsg+'\n')
            lres[name+'-'+wavelength+'-'+result] = [ei]+res
    return lres


def process_caribu(scene, meteo, outdir = None):
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)


    for index, row in meteo.iterrows():
        time, globalirr, diffuseirr, temperature = row
        print(time, globalirr, diffuseirr, temperature)
        sun, sky= sun_sky_sources(ghi = globalirr, dhi = diffuseirr, **localisation)
        raw, res = caribu(scene, sun, sky)
        scene.plot(raw['PAR']['Ei'], minval=0, maxval=100)
        print(res)



def main():
    # read the meto
    meteo = read_meteo()

    # a digitized mango tree
    scene = generate_plots()

    csScene = toCaribuScene(scene,OPTPROP)

    process_caribu(csScene, meteo)

if __name__ == '__main__':
    main()