
import pandas
from openalea.plantgl.all import *
from openalea.plantgl.scenegraph.colormap import PglMaterialMap

from alinea.astk.sun_and_sky import sun_sky_sources, sun_sources
import os, sys

from importlib import reload
import generateplot; reload(generateplot)
from generateplot import *
import time, datetime, pytz
from os.path import join
import numpy as np

DEBUG = False
RESOLUTION = None #0.1

localisation={'latitude':-7.473411, 'longitude':112.462442, 'timezone': 'Asia/Jakarta'}
north = -180


def lightsRepr(sun, sky, dist = 40, spheresize = 0.8):
  import openalea.plantgl.all as pgl
  from openalea.plantgl.light import azel2vect
  from openalea.plantgl.scenegraph.colormap import PglMaterialMap
  sun_m = sun[1], sun[0], sun[2]
  directions = list(zip(*sun_m)) 
  if not sky is None:
    sky_m = sky[1], sky[0], sky[2]
    directions += list(zip(*sky_m))
  s = pgl.Scene()
  sp = Sphere(spheresize)
  cmap = PglMaterialMap(min(0,min([i for az,el,i in directions])),max([i for az,el,i in directions]))
  for az,el,i in directions:
    if i > 0:
        dir = -azel2vect(az, el, north=-north)
        s += pgl.Shape(pgl.Translated(dir*dist+Vector3(22.5,0,0),sp),cmap(i))
  return s

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
    raw, agg = scene.run(direct=True, infinite = False, split_face = False, screen_resolution=RESOLUTION)
    print('made in', time.time() - t)
    if view : 
        Ei = raw['PAR']['Ei']
        maxei = max([max(v) for pid,v in Ei.items()])
        scene.plot(Ei, minval=0, maxval=maxei)
        cm = PglMaterialMap(0, maxei)
        Viewer.add(lightsRepr(sun, sky)+cm.pglrepr())
    return raw, agg, grid_values(raw['PAR']['Ei'][SOIL])

def mplot( scene, scproperty, minval = None, display = True):
    from openalea.plantgl.scenegraph import Scene, Shape
    from openalea.plantgl.gui import Viewer
    nscene = Scene()
    cm = PglMaterialMap(0, max(scproperty.values()))
    for sh in scene:
        if minval is None or scproperty[sh.id] > minval :
            nscene.add(Shape(sh.geometry, cm(scproperty[sh.id]), sh.id))
    if display:
        Viewer.display(nscene+cm.pglrepr())
    return nscene

def toPglScene(scene):
    t = time.time()
    print ('Convert scene for plantgl')
    tes = Tesselator()
    tscene = Scene()
    id = 0
    idmap = {}
    for sh in scene:
        sh.apply(tes)
        for tr in tes.result.indexList:
            tscene.add(Shape(TriangleSet([tes.result.pointList[tri] for tri in tr], [list(range(3))]),id=id))
            idmap[id] = sh.id
            id += 1
    print('done in', time.time() - t)
    return tscene, idmap

def _agregate(df, idmap):
    agg = df.aggregate(axis = 1 )

def grid_values(irradiances):
    irr = [irradiances[i]+irradiances[i+1] for i in range(0,len(irradiances),2)]
    colsrows =  groundids()
    cols = [c for c,r in colsrows]
    rows = [r for c,r in colsrows]
    return pandas.DataFrame({'column':cols,'rows':rows,'irradiance':irr})


def plantgllight(scene, sun, sky,  view = False):
    from openalea.plantgl.light import scene_irradiance
    tscene, idmap = toPglScene(scene)    
    print('start plantgl ...')
    t = time.time()
    # permute az, el, irr to el, az, irr
    sun_m = sun[1], sun[0], 1 #sun[2]
    sky_m = sky[1], sky[0], 0 #sky[2]
    directions = list(zip(*sun_m)) + list(zip(*sky_m))
    # For plantgl, north is given in counter clockwise. caribu is given in clockwise.
    defm = scene_irradiance(tscene, directions, horizontal=True, screenresolution=RESOLUTION, scene_unit='m', north = -north)
    print('made in', time.time() - t)
    defm.insert(2,'type',idmap)
    if view:
        d = dict(list(zip(list(defm.index), defm['irradiance'])))
        mplot(tscene, d)

    gridirradiances = defm[defm['type']==SOIL]['irradiance'].tolist()
    return defm, defm.groupby('type').sum(), grid_values(gridirradiances)

tz = pytz.timezone(localisation['timezone'])
def date(month, day, hour, minutes=0, seconds = 0):
    return pandas.Timestamp(datetime.datetime(2023, month, day, hour, minutes, seconds), tz=tz)

from datetime import timedelta

""" Ne considerez que du '17-May' au '31-Oct' """
def process_light(heigth=0.5, orientation = 50, mindate = date(5,17,0), maxdate = None, timestep = timedelta(days=0, hours = 1, minutes = 0), diffuse = 0, usecaribu = True, view = True, outdir = None):
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)
    
    if maxdate is None:
        maxdate = mindate

    # the scene with the panels
    scene = generate_plots(heigth, orientation)+ground()

    
    if usecaribu :
        scene = toCaribuScene(scene,OPTPROP)

    currentdate = mindate

    results = []
    while currentdate <= maxdate:
            print(currentdate)
            sun, sky= sun_sky_sources(ghi = 1, dhi = diffuse, dates=currentdate, **localisation)
            sun = sun[0], sun[1], np.ones(len(sun[0]))*(1-diffuse)
            if diffuse == 0:
                sky = None
            if usecaribu :
                result = caribu(scene, sun, sky, view=view)
            else:
                result = plantgllight(scene, sun, sky, view=view)
            _,_,gvalues = result
            if outdir:
                gvalues.to_csv(join(outdir,'grid_'+currentdate.strftime('%Y-%m-%d-%H-%M')+'.csv'),sep='\t')
            results.append((currentdate,gvalues))
            currentdate += timestep
    return results

if __name__ == '__main__':
    hour = 13
    month = 12
    results = process_light(mindate = date(month,21,hour), outdir='result')
    print(results)