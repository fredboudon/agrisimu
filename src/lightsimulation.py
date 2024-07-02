
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
north = 0


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
        minei = min([min(v) for pid,v in Ei.items()])
        maxei = max([max(v) for pid,v in Ei.items()])
        pglscene, _ = scene.plot(Ei, minval=minei, maxval=maxei, display=False)
        cm = PglMaterialMap(minei, maxei)
        pglscene +=  lightsRepr(sun, sky)+cm.pglrepr()
        #Viewer.add(lightsRepr(sun, sky)+cm.pglrepr())
    else:
        pglscene = None
    return raw, agg, grid_values(raw['PAR']['Ei'][SOIL]), pglscene

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
    return pandas.DataFrame({'column':cols,'row':rows,'irradiance':irr})

def matrix_values(df):
    res = np.zeros((max(df['row'])+1,max(df['column'])+1))
    for index, row in df.iterrows():
        res[int(row['row']),int(row['column'])] = float(row['irradiance'])
    return res


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

    gridirradiances = defm[defm['type']==SOIL]['irradiance'].tolist()
    return defm, defm.groupby('type').sum(), grid_values(gridirradiances), scene

tz = pytz.timezone(localisation['timezone'])
def date(month, day, hour, minutes=0, seconds = 0):
    return pandas.Timestamp(datetime.datetime(2023, month, day, hour, minutes, seconds), tz=tz)

from datetime import timedelta

""" Ne considerez que du '17-May' au '31-Oct' """
def process_light(nb = 5,panelheight = 0.5, height=0.5,  mindate = date(5,17,0), maxdate = None, timestep = timedelta(days=0, hours = 1, minutes = 0), diffuse = 0, usecaribu = True, view = True, outdir = None):
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)
    
    if maxdate is None:
        maxdate = mindate

    # the scene with the panels
    scene = generate_plots2(nb, panelheight, height)
    scene += ground()

    
    if usecaribu :
        scene = toCaribuScene(scene,OPTPROP)

    currentdate = mindate

    results = []
    while currentdate <= maxdate:
            print(currentdate)
            sun, sky= sun_sky_sources(ghi = 1, dhi = diffuse, dates=currentdate, **localisation)
            # sun = sun[0], sun[1], np.ones(len(sun[0]))*(1-diffuse)
            if diffuse == 0:
                sky = None
            if usecaribu :
                result = caribu(scene, sun, sky, view=view)
            else:
                result = plantgllight(scene, sun, sky, view=view)
            _,_,gvalues, sc = result
            resarray = matrix_values(gvalues)
            import matplotlib.pyplot as plt
            plt.imshow(resarray, cmap='jet',vmin = 0)
            plt.show(block=False)
            fname = join(outdir,'grid__'+str(nb)+'_'+str(int(panelheight*100))+'_'+str(int(height*100))+'__'+currentdate.strftime('%Y-%m-%d-%H-%M'))
            if outdir:
                gvalues.to_csv(fname+'.csv',sep='\t')
                if view:
                    sc.save(fname+'.bgeom')
                plt.savefig(fname+'.png')
            results.append((currentdate,gvalues))
            currentdate += timestep
    return results

if __name__ == '__main__':
    day=1
    month =7
    results =  process_light(height=0.1, panelheight = 0.2, nb = 5, mindate = date(month,day,6) , maxdate = date(month,day,17), timestep = timedelta(days=0, hours = 0, minutes = 20), outdir='result', usecaribu=True)
    print(results)
    #process_light(heigth=1,orientation=1,mindate = date(month,day,6) , maxdate = date(month,day,17), outdir='result')
    #process_light(heigth=1.5,orientation=1,mindate = date(month,day,6) , maxdate = date(month,day,17), outdir='result')
    #process_light(heigth=1.8,orientation=1,mindate = date(month,day,6) , maxdate = date(month,day,17), outdir='result')
    #process_light(heigth=2,orientation=1,mindate = date(month,day,6) , maxdate = date(month,day,17), outdir='result')
