
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

DEBUG = False
RESOLUTION = 0.1

localisation={'latitude':42.77, 'longitude':2.86, 'timezone': 'Europe/Paris'}
north = -39.9
# -90 


def lightsRepr(lights, dist = 40, spheresize = 0.8):
  import openalea.plantgl.all as pgl
  from openalea.plantgl.light import azel2vect
  from openalea.plantgl.scenegraph.colormap import PglMaterialMap
  s = pgl.Scene()
  mmax = max([i for i, pos in lights])
  cmap = PglMaterialMap(min(0,min([i for i, pos in lights])),mmax)
  for i, dir in lights:
    if i > 0:
        s += pgl.Shape(pgl.Translated(-pgl.Vector3(dir)*dist,pgl.Sphere(spheresize*(0.5+i/(2*mmax)))),cmap(i))
  return s


def toCaribuScene(scene, opt_prop) :
    from alinea.caribu.CaribuScene import CaribuScene
    t = time.time()
    print ('Convert scene for caribu')
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
    sc = Scene()
    if view : 
        Ei = raw['PAR']['Ei']
        maxei = max([max(v) for pid,v in Ei.items()])
        sc, _ = scene.plot(Ei, minval=0, maxval=maxei)
        cm = PglMaterialMap(0, maxei)
        sc += cm.pglrepr()
        sc += lightsRepr(light)
    return raw, agg, grid_values(raw['PAR']['Ei'][SOIL]), sc

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
    irr = [(irradiances[i]+irradiances[i+1])/2 for i in range(0,len(irradiances),2)]
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
    sun_m = sun[1], sun[0], sun[2]
    sky_m = sky[1], sky[0], sky[2]
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
def sdate(month, day, hour):
    return pandas.Timestamp(datetime.datetime(2023, month, day, hour, 0, 0)) #, tz=tz)

""" Ne considerez que du '17-May' au '31-Oct' """
def process_light(mindate = sdate(5,1,0), maxdate = None, usecaribu = True, view = True, outdir = None):
    from datetime import date
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

    if maxdate is None and mindate != None:
        if mindate.hour == 0 and mindate.minute == 0:
            from copy import copy
            maxdate = date(mindate.year, mindate.month, mindate.day, 23, 59)
        else:
            maxdate = mindate
    
    currentdate = mindate

    # read the meteo
    meteo,  params = read_meteo( )

    if params is None:
        class NoneIter:
            def __init__(self):
                pass
            def __iter__(self):
                return self
            def __next__(self):
                return None, None
        paramiter = NoneIter()
    else:
        paramiter = params.iterrows()

    # an agrivoltaic scene (generate plot)
    height = 0.0
    scene = None
    
    initdate = sdate(1,1,0)

    results = []
    for (cdate, values), (cdate2, param) in zip(meteo.iterrows(), paramiter):
        ghi, dhi = values
        if mindate is None or (cdate >= mindate and cdate < maxdate):
            if param is None:
                if scene is None:
                    scene = generate_plots()+ground(height)
                    if usecaribu :
                        scene = toCaribuScene(scene,OPTPROP)

            else:
                param = param.tolist()
                if param != lastparam:
                    print('Generate scene')
                    scene = generate_plots(*params)+ground(height)
                    lastparam = param
                    if usecaribu :
                        scene = toCaribuScene(scene,OPTPROP)
            sun, sky= sun_sky_sources(ghi = ghi, dhi = min(ghi,dhi), dates=cdate, **localisation)
            if ghi > 0:
                if usecaribu :
                    result = caribu(scene, sun, sky, view=view)
                else:
                    result = plantgllight(scene, sun, sky, view=view)
                _,_,gvalues, sc = result
                if outdir:
                    gvalues.to_csv(join(outdir,'grid_'+str(height).replace('.','_')+'_'+cdate.strftime('%Y-%m-%d-%H-%M')+'.csv'),sep='\t')
                    if view:
                        sc.save(join(outdir,'grid_'+str(height).replace('.','_')+'_'+cdate.strftime('%Y-%m-%d-%H-%M')+'.bgeom'))
                results.append((cdate,gvalues))                     
            print(cdate, ghi)

    return results

if __name__ == '__main__':
    # date(month,day,hour)
    results = process_light(sdate(8,15,14), sdate(8,15,15), outdir='result', view=True)
    print(results)

