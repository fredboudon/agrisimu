from openalea.plantgl.all import (
    AxisRotated, Color3, Cylinder, Font, Material, PglMaterialMap, Polyline, QuadSet,
    Scene, ScreenProjected, Shape, Text, Translated, Vector3, Viewer, discretize,
)
from math import radians
from pathlib import Path

from .paths import DATA_DIR, OUTPUT_DIR

PANELS, POLES, WIRES, SOIL, SENSORS = 1, 2, 3, 4, 5
# Reflectance_Up, Transmittance_Up, Reflectance_Down, Transmittance_Down

# Lower-left support post at 1.5 m on the X axis and 1 m on the Y axis
# square layout                3           2
# next       3           8

def agristructure(initial_offset=(-2,0)):
#def agristructure(initial_offset=(0,0)):
    panel = QuadSet(
        [(0, 0, 0), (1.0, 0, 0), (1.0, 1.93, 0), (0, 1.93, 0)],
        [list(range(4))],
    )

    panel3 = [
        Translated(0, 0, 5.12, AxisRotated(Vector3.OX, radians(15), p))
        for p in [panel, Translated(1.075, 0, 0, panel), Translated(2.15, 0, 0, panel)]
    ]

    row = [Translated(7.5 * i, 0, 0, p) for i in range(6) for p in panel3]

    panelmatrix = [
        Translated(3.7 if j % 2 == 0 else 7.45, 3 * j, 0, geom)
        for j in range(5)
        for geom in row
    ]

    scene = Scene([Shape(panel, id=PANELS) for panel in panelmatrix])

    borderpole = Cylinder(0.125, 6.81, solid=False)

    # 7.58 on the left
    # 7.02 on the right
    leftangle = 7.58
    rightangle = 7.02

    leftborderpoles = [
        Translated(3.4, 0.96 + 3 * i, 0, p)
        for p in [
            AxisRotated(
                Vector3.OY,
                radians(-30),
                AxisRotated(Vector3.OX, radians(leftangle), borderpole),
            ),
            AxisRotated(
                Vector3.OY,
                radians(-30),
                AxisRotated(Vector3.OX, radians(-rightangle), borderpole),
            ),
        ]
        for i in range(5)
    ]

    rightborderpoles = [
        Translated(48.4, 0.96 + 3 * i, 0, p)
        for p in [
            AxisRotated(
                Vector3.OY,
                radians(30),
                AxisRotated(Vector3.OX, radians(leftangle), borderpole),
            ),
            AxisRotated(
                Vector3.OY,
                radians(30),
                AxisRotated(Vector3.OX, radians(-rightangle), borderpole),
            ),
        ]
        for i in range(5)
    ]

    borderpoles = leftborderpoles + rightborderpoles

    centralpole = Cylinder(0.125 / 2, 5.9, solid=False)

    leftcentralpoles = [
        Translated(18.4, 0.96 + 3 * i, 0, p)
        for p in [
            AxisRotated(Vector3.OX, radians(leftangle), centralpole),
            AxisRotated(Vector3.OX, radians(-rightangle), centralpole),
        ]
        for i in range(5)
    ]

    rightcentralpoles = [
        Translated(33.4, 0.96 + 3 * i, 0, p)
        for p in [
            AxisRotated(Vector3.OX, radians(leftangle), centralpole),
            AxisRotated(Vector3.OX, radians(-rightangle), centralpole),
        ]
        for i in range(5)
    ]

    scene += Scene(
        [
            Shape(pole, id=POLES)
            for pole in borderpoles + leftcentralpoles + rightcentralpoles
        ]
    )
    for sh in scene:
        if type(sh.geometry) == Translated:
            sh.geometry.translation = sh.geometry.translation + Vector3(initial_offset[0],initial_offset[1], 0)
        else :
            sh.geometry = Translated(initial_offset[0],initial_offset[1],0, sh.geometry)
    return scene


def retrieve_panel_projection(dfactor = 0.5):
    from openalea.plantgl.all import discretize
    s = agristructure()
    panels = [discretize(sh.geometry) for sh in s if sh.id == PANELS]
    projection = []
    for panel in panels:
        for idx in panel.indexList:
            box = []
            for id in idx:
                v = panel.pointList[id]
                box.append((v.x, v.y))
            projection.append(box)
    return projection

##### TILES FOR RAY TRACING
########### COLORS FOR FLOORS
ricecol = Material("#88AA59", Color3(80,100,45))
groundcol = Material("#B68354", Color3(135,100,80))
sensorcol = Material("#94B5DA", Color3(148,181,218))

NBCOL = 100
NBLIG = 30
MAPLENGTH = 50.0  # m
MAPWIDTH = 15.0   # m

def sensorsids():
    return [(col, rank)  for rank in range(NBLIG) for col in range(NBCOL)]

IDDECALPOWER = 4
IDDECAL = pow(10, IDDECALPOWER)

def position2id(col, rank):
    return int(rank*pow(10, IDDECALPOWER*2) + col*IDDECAL)

def id2position(id):
    col = (id // IDDECAL) % IDDECAL
    rank = id // pow(10,IDDECALPOWER*2)
    return int(col), int(rank)

def sensorpositions(height = 0, filter = None):
    WCOL = MAPLENGTH / NBCOL
    WROW = MAPWIDTH / NBLIG
    if filter is None:
        filter = lambda v : v
    return [(position2id(col, rank),(WCOL*col+WCOL/2,WROW*rank+WROW/2,height)) for col, rank in filter(sensorsids())]

def sensorgeometry(height = 0, filter = None):
    WCOL = MAPLENGTH / NBCOL
    WROW = MAPWIDTH / NBLIG
    tile = QuadSet([(-WCOL/2,-WROW/2,0),(WCOL/2,-WROW/2,0),(WCOL/2,WROW/2,0),(-WCOL/2,WROW/2,0)], [list(range(4))])
    floor = [Shape(Translated(position,tile), groundcol, id) for id, position in sensorpositions(height, filter=filter)]
    ########### GROUND
    return  Scene(floor)

def plot_light_source(meteo):
    from openalea.plantgl.light import LightEstimator
    from openalea.astk.sky_sources import caribu_light_sources
    from .localization import localisation
    scene = agristructure()+sensorgeometry()
    l = LightEstimator(scene)
    l.localize(name = 'Camargue', **localisation)
    #l.add_sun(meteo.index, 1)
    assert all([col in meteo.columns for col in ['ghi', 'dhi', 'elevation', 'azimuth']])
    assert len(meteo.index) > 0
    for cdate, row in meteo.iterrows():
        globalirr, diffuseirr = row['ghi'],row['dhi']
        #irr, dir = caribu_light_sources([(row['elevation'], -row['azimuth'],  ghi)],[], north = l.north)[0]
        #l.add_light_from_vector(f"sun_{cdate.strftime('%Y%m%d_%H%M')}", Vector3(dir),  1, horizontal=True, date=cdate, type='SUN')
        l.add_light(f"sun_{cdate.strftime('%Y%m%d_%H%M')}", row['elevation'], row['azimuth'],  globalirr-diffuseirr, horizontal=True, date=cdate, type='SUN')
        l.add_sky(irradiance = diffuseirr)
    l.plot()
    return l


def project_group_from_file(geometry = None, fname = DATA_DIR / 'clusters.csv'):
    import pandas as pd
    data = pd.read_csv(fname)
    idmap = { idx:i for i,idx in enumerate(sensorsids()) }
    cmap = { 1 : (44, 87, 93), 2 : (195, 214, 155), 3 : (85, 143, 156), 4 : (92, 92, 50) }
    colors = [None for i in idmap]
    for i, row in data.iterrows():
        col, row, group = row['Col'],row['Row'],row['group_clearsky']
        colors[idmap[(col, row)]] = Material(cmap[group], diffuse=1)
    return Scene([Shape(sh.geometry, col if col else Material(transparency=1), sh.id) for sh, col in zip(geometry, colors)])


def project_cluster(geometry = None, borderline = False):
    from .data_util import get_clusters
    clusters = get_clusters(center=False)
    cmap = { 1 : (44, 87, 93), 2 : (195, 214, 155), 3 : (85, 143, 156), 4 : (92, 92, 50) }
    colors = [ Material(transparency=1) if not pid in clusters else Material(cmap[clusters[pid]], diffuse=1) for pid in sensorsids()]
    res = Scene([Shape(sh.geometry, col, sh.id) for sh, col in zip(geometry, colors)])
    if borderline:
        clusters = get_clusters(center=True)
        border = [ Translated(0,0,0.01,Polyline(discretize(geom.geometry).pointList, width=5)) for pid, geom in zip(sensorsids(), geometry) if pid in clusters ]
        for line in border:
            line.geometry.pointList.append(line.geometry.pointList[0])
        res += Scene([Shape(b, Material("RED", Color3(255,0,0))) for b in border])
    return res

def project_data(data, vmin=None, vmax=None, cmap='jet', *, geometry=None):
    if vmin is None:
        vmin = min(data.values())
    if vmax is None:
        vmax = max(data.values())
    cmap = PglMaterialMap(name=cmap, minvalue=vmin, maxvalue=vmax)
    colors = [ Material(transparency=1) if not pid in data else cmap(data[pid]) for pid in sensorsids()]
    geometry = sensorgeometry() if geometry is None else geometry
    return Scene([Shape(sh.geometry, col, sh.id) for sh, col in zip(geometry, colors)])

#IRR_PER_PLOT = {}
def project_irradiance(geometry, date, datarep=OUTPUT_DIR / 'result' / 'clear_sky', column = 'irradiance'):
    from agrisimu.data_util import get_irradiances_per_plot, meteo_select_dates, timeline_select_dates
    irr_per_plot = get_irradiances_per_plot(datarep)
    irr_per_plot = { pid : meteo_select_dates(irr_per_plot[pid], date) for pid in sensorsids() }
    vmin = min([irr_per_plot[pid][column].min() for pid in sensorsids()])
    vmax = max([irr_per_plot[pid][column].max() for pid in sensorsids()])
    for date in irr_per_plot[sensorsids()[0]].index:
        data = { pid : irr_per_plot[pid].loc[date][column] for pid in sensorsids() }
        yield date, project_data(data, vmin=vmin, vmax=vmax, geometry=geometry)

def create_animation(sceneiterator, outputfolder=OUTPUT_DIR / 'animation', prefix = 'frame'):
    import os
    if not outputfolder is None and not os.path.exists(outputfolder):
        os.makedirs(outputfolder, exist_ok=True)
    nbtimepoints = len(next(sceneiterator))
    print(f"Number of time points : {nbtimepoints}")
    Viewer.frameGL.setSize(1200, 600)
    for i, (date, scene) in enumerate(sceneiterator):
        print(date)
        Viewer.camera.lookAt(Vector3(Vector3.Spherical(60,radians(-110+40*i/nbtimepoints), radians(-50))),(0,0,0))
        Viewer.display(Scene([Shape(Translated(-25,-7.5,0, sh.geometry), sh.appearance, sh.id) for sh in scene+
                       Scene([Shape(sh.geometry, Material((0,0,0), transparency = 0.5), sh.id) for sh in agristructure()]+
                             [Shape(ScreenProjected(Text(str(date),(0.7,-0.4,0), fontstyle=Font('',size=15))),Material((0,0,0)))])]))
        if not outputfolder is None:
            Viewer.frameGL.saveImage(os.path.join(outputfolder, f"{prefix}_{date.strftime('%Y%m%d_%H%M')}.png"))
        from PyQt5.QtCore import QCoreApplication
        app = QCoreApplication.instance()
        if app is not None:
            app.processEvents()

def projet_day_irradiance( data=OUTPUT_DIR / 'result' / 'weather2023', date = '2023/06/20', column = 'irradiance'):
    from agrisimu.data_util import get_irradiances_per_plot, meteo_select_dates
    if isinstance(data, (str, Path)):
        data = get_irradiances_per_plot(data)
    data = { pid : meteo_select_dates(data[pid], date) for pid in sensorsids() }
    vmin = min([data[pid][column].min() for pid in sensorsids()])
    vmax = max([data[pid][column].max() for pid in sensorsids()])
    yield data[sensorsids()[0]].index
    for date in data[sensorsids()[0]].index:
        dataval = { pid : data[pid].loc[date][column] for pid in sensorsids() }
        yield date, project_data(dataval, vmin=vmin, vmax=vmax)


def projet_season_irradiance(data=OUTPUT_DIR / 'result' / 'weather2023', hour = "12:00", column = 'irradiance'):
    from agrisimu.data_util import get_irradiances_per_plot, meteo_select_dates
    import pandas as pd
    if isinstance(data, (str, Path)):
        data = get_irradiances_per_plot(data)
    data = { pid : data[pid][data[pid].index.time == pd.to_datetime(hour).time()] for pid in sensorsids() }
    vmin = min([data[pid][column].min() for pid in sensorsids()])
    vmax = max([data[pid][column].max() for pid in sensorsids()])
    yield data[sensorsids()[0]].index
    for date in data[sensorsids()[0]].index:
        dataval = { pid : data[pid].loc[date][column] for pid in sensorsids() }
        yield date, project_data(dataval, vmin=vmin, vmax=vmax)


if __name__ == "__main__":
    import datetime

    from .meteo import setup_meteo

    clearsky = setup_meteo().clearsky200
    plot_light_source(clearsky[clearsky.index.date == datetime.date(2023, 7, 20)])
