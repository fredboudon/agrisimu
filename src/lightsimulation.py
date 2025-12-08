
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
    raw, agg = scene.run(direct=True, infinite = False, split_face = False, screen_resolution=RESOLUTION)
    print('made in', time.time() - t)
    sc = None
    if view : 
        Ei = raw['PAR']['Ei']
        maxei = max([max(v) for pid,v in Ei.items()])
        sc, _ = scene.plot(Ei, minval=0, maxval=maxei)
        cm = PglMaterialMap(0, maxei)
        sc += cm.pglrepr()
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
def date(month, day, hour):
    return pandas.Timestamp(datetime.datetime(2023, month, day, hour, 0, 0), tz=tz)

""" Ne considerez que du '17-May' au '31-Oct' """
def process_light(mindate = date(5,1,0), maxdate = date(11, 1,0), usecaribu = True, view = True, outdir = None):
    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

    # read the meteo
    meteo = read_meteo()

    # an agrivoltaic scene (generate plot)
    height = 0.5
    scene = generate_plots()+ground(height)
    
    if usecaribu :
        scene = toCaribuScene(scene,OPTPROP)

    initdate = date(1,1,0)

    results = []
    for index, row in meteo.iterrows():
        time, globalirr, diffuseirr, temperature = row
        cdate = initdate+datetime.timedelta(seconds=time)
        if cdate >= mindate and cdate < maxdate:
            print(time, cdate,globalirr, diffuseirr, temperature)
            sun, sky= sun_sky_sources(ghi = globalirr, dhi = diffuseirr, dates=cdate, **localisation)
            # dhi=none if no diffuse data
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
    return results

if __name__ == '__main__':
    # date(month,day,hour)
    results = process_light(date(11,1,0), date(11,3,0), outdir='result', view=False)
    print(results)

