
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
from datetime import timedelta
import matplotlib.pyplot as pp

DEBUG = False
RESOLUTION = None #0.1

north = -90



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
        s += pgl.Shape(pgl.Translated(dir*dist,sp),cmap(i))
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
        # orientation: (float)  the angle (deg, positive clockwise) from X+ to North (default: 0)
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
        minei = min([min(v) for pid,v in Ei.items()])
        maxei = max([max(v) for pid,v in Ei.items()])
        pglscene, _ = scene.plot(Ei, minval=minei, maxval=maxei, display=False)
        cm = PglMaterialMap(minei, maxei)
        pglscene +=  lightsRepr(sun, sky)+cm.pglrepr()
        #Viewer.add(lightsRepr(sun, sky)+cm.pglrepr())
    else:
        pglscene = None
    return raw, agg, [agg['PAR']['Ei'][sensor] for sensor in SENSORS if sensor in agg['PAR']['Ei']] , pglscene

def mplot( scene, scproperty, minval = None, display = False):
    from openalea.plantgl.scenegraph import Scene, Shape
    from openalea.plantgl.gui import Viewer
    nscene = Scene()
    cm = PglMaterialMap(0, max(scproperty.values()))
    for sh in scene:
        if minval is None or scproperty[sh.id] > minval :
            nscene.add(Shape(sh.geometry, cm(scproperty[sh.id]), sh.id))
    if display:
        Viewer.display(nscene+cm.pglrepr())
    return nscene+cm.pglrepr()

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
    if sun:
       sun_m = sun[1], sun[0], sun[2]
    else:
        sun_m = []
    if sky:
       sky_m = sky[1], sky[0], sky[2]
    else:
       sky_m = []
    directions = list(zip(*sun_m)) + list(zip(*sky_m))
    # For plantgl, north is given in counter clockwise. caribu is given in clockwise.
    defm = scene_irradiance(tscene, directions, horizontal=True, screenresolution=RESOLUTION, scene_unit='m', north = -north)
    print('made in', time.time() - t)
    defm.insert(2,'type',idmap)
    if view:
        d = dict(list(zip(list(defm.index), defm['irradiance'])))
        scene = mplot(tscene, d)
    else:
        scene = None

    # gridirradiances = defm[defm['type']==SOIL]['irradiance'].tolist()
    print(defm)
    totirr = defm['area']*defm['irradiance']
    defm2 = defm[['type','area']]
    defm2['irradiance'] = totirr
    agg = defm2.groupby('type').sum()
    agg['irradiance'] /= agg['area']
    # agg = defm.groupby('type').sum()
    print(agg)
    return defm, agg, agg['irradiance'][SENSORS] , scene

def process_light(name, genscene, inputdata, mindate = None, maxdate = None, localisation = None, outdir= None, usecaribu = True, view = True):
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)
    
    if maxdate is None:
        if mindate.hour == 0 and mindate.minute == 0:
            from copy import copy
            maxdate = date(mindate.year, mindate.month, mindate.day, 23, 59, localisation=localisation)
        else:
            maxdate = mindate
    
    currentdate = mindate

    resultdate = []
    resultirr = []
    lastparam = None
    meteo, refsensors, params = inputdata
    scene = None
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
    finalresult = []
    for (cdate, values), (cdate2, param) in zip(meteo.iterrows(), paramiter):
        ghi, dhi = values
        if mindate is None or (cdate >= mindate and cdate < maxdate):
            if param is None:
                if scene is None:
                    scene = genscene()
                    if usecaribu :
                        scene = toCaribuScene(scene,OPTPROP)

            else:
                param = param.tolist()
                if param != lastparam:
                    print('Generate scene')
                    scene = genscene(*param)
                    lastparam = param
                    if usecaribu :
                        scene = toCaribuScene(scene,OPTPROP)
            sun, sky= sun_sky_sources(ghi = ghi, dhi = min(ghi,dhi), dates=cdate, **localisation)
            if ghi > 0:
                if usecaribu :
                    result = caribu(scene, sun, sky, view=view)
                else:
                    result = plantgllight(scene, sun, sky, view=view)
                _,_, sensors, sc = result
                if outdir :
                     sc.save(join(outdir,'scene_'+str(cdate)+'.bgeom'))
            else:
                sensors = [0 for i in range(len(refsensors.columns))]
            resultirr.append(sensors)
            resultdate.append(cdate)
            print(cdate, ghi, dhi, sensors)
            for i, s in enumerate(sensors,1):
                finalresult.append([name, cdate, i, s])

    resultirr = pandas.DataFrame(
                dict([("vsensor"+str(i), [v[i] for v in resultirr]) for i in range(len(resultirr[0]))]), 
                     index=resultdate)
    result = pandas.concat((meteo, resultirr,refsensors), axis=1)
    result = result.loc[resultirr.index]

    fresult = pandas.DataFrame([]
                dict([(name, [v[i] for v in finalresult]) for i, name in enumerate(['site', 'date', 'sensor', 'r_intercepted'])]))

    return result, fresult



def date(year, month, day, hour, minutes=0, seconds = 0, localisation = None):
    return pandas.Timestamp(datetime.datetime(year, month, day, hour, minutes, seconds), tz=pytz.timezone(localisation['timezone']))

def get_days(df):
    return list(sorted(set(map(lambda d: str(d.date()),df.index.to_pydatetime()))))

EDF, TOTAL, VALOREM = range(3)
TEST = VALOREM
if __name__ == '__main__':
    if TEST == EDF:
        scene, localisation, meteo = edf_system()
        results, fresult =  process_light('edf',scene,meteo,mindate = date(2023,10,1,0, localisation=localisation) , 
                                   maxdate = date(2023,10,1,23,50, localisation=localisation), 
                                   localisation=localisation,
                                   outdir='result_edf', usecaribu=True)
        fresult.to_csv('edf_result.csv')
    elif TEST == TOTAL:
        scene, localisation, meteo = total_system()
        results, fresult =  process_light('total',scene,meteo,mindate = date(2023,4,4,0, localisation=localisation) , 
                                   maxdate = date(2023,4,4,23,50, localisation=localisation), 
                                   localisation=localisation,
                                   outdir='result_total', usecaribu=True)
        fresult.to_csv('total_result.csv')
    elif TEST == VALOREM:
        scene, localisation, meteo = valorem_system()
        # print(get_days(meteo))
        # ['2023-06-22', '2023-07-03', '2023-07-14', '2023-09-25', '2023-10-27', '2023-11-05']
        results, fresult =  process_light('valorem', scene,meteo,localisation=localisation, mindate = date(2023,7,3,0,0, localisation=localisation),
                                   outdir='result_valorem', usecaribu=True)
        fresult.to_csv('valorem_result.csv')


    print(results)
    results.plot()
    pp.show()
