
import pandas
from openalea.plantgl.all import *
from alinea.astk.sun_and_sky import sun_sky_sources, sun_sources
import os, sys

from importlib import reload
import generateplot; reload(generateplot)
from generateplot import *
import time, datetime, pytz
from os.path import join

DEBUG = False

localisation={'latitude':43.734286, 'longitude':4.570565, 'timezone': 'Europe/Paris'}
north = -90

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

# on considere le nord en Y.
def caribu(scene, sun = None, sky = None, view = False, debug = False):
    from alinea.caribu.light import light_sources
    print('start caribu...')
    t = time.time()
    print('Create light source', end=' ')
    light = []
    if not sun is None:
        light += light_sources(*sun, orientation = north) 
    if not sky is None:
        light += light_sources(*sky, orientation = north)
    print('... ',len(light),' sources.')
    scene.setLight(light)
    print('Run caribu')
    #raw, agg = scene.run(direct=False, infinite = True, split_face = True, d_sphere = D_SPHERE)
    raw, agg = scene.run(direct=True, infinite = False, split_face = False)
    print('made in', time.time() - t)
    if view : 
        scene.plot(raw['PAR']['Ei'])
    return raw, agg


def normalize_energy(lights):
    el,az,ei = lights
    sumei = ei.sum()
    lights = el, az, ei / sumei
    return lights, sumei




""" Ne considerez que du '17-May' au '31-Oct' """
def process_caribu(scene, meteo, display = True, outdir = None):
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

    tz = pytz.timezone(localisation['timezone'])
    initdate = pandas.Timestamp(datetime.datetime(2023, 1,1,0,0,0), tz=tz)
    mindate = pandas.Timestamp(datetime.datetime(2023, 5,17,0,0,0), tz=tz)
    maxdate = pandas.Timestamp(datetime.datetime(2023, 11, 1,0,0,0), tz=tz)

    mindate = pandas.Timestamp(datetime.datetime(2023, 5,17,9,0,0), tz=tz)
    maxdate = pandas.Timestamp(datetime.datetime(2023, 5,17,10,0,0), tz=tz)

    for index, row in meteo.iterrows():
        time, globalirr, diffuseirr, temperature = row
        cdate = initdate+datetime.timedelta(seconds=time)
        if cdate >= mindate and cdate < maxdate:
            print(time, cdate,globalirr, diffuseirr, temperature)
            sun, sky= sun_sky_sources(ghi = globalirr, dhi = diffuseirr, dates=cdate, **localisation)
            print(sky)
            print(sun)
            raw, res = caribu(scene, sun, sky)
            print(res)
            Ei = raw['PAR']['Ei']
            maxei = max([max(v) for pid,v in Ei.items()])
            print(maxei)
            if display:
                scene.plot(Ei, minval=0, maxval=maxei)



def main():
    # read the meto
    meteo = read_meteo()

    # a digitized mango tree
    scene = generate_plots()+ground()

    csScene = toCaribuScene(scene,OPTPROP)

    process_caribu(csScene, meteo)

if __name__ == '__main__':
    main()