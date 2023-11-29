from openalea.plantgl.all import *
from math import *

PANELS, POLES, WIRES, SOIL, SENSORS = 1,2,3,4,5
# Reflectance_Up, Transmittance_Up, Reflectance_Down, Transmittance_Down
OPTPROP = { 'PAR' :{ PANELS : (0.01,0.01,0.01,0.01), 
                     POLES   : (0.01,0.01,0.01,0.01),
                     WIRES  : (0.01,0.01,0.01,0.01),
                     SOIL   : (0.01,0.01,0.01,0.01)}}


def generate_plots():
    panel = QuadSet([(0,0,0),(1.0,0,0),(1.0,1.93,0),(0,1.93,0)], [list(range(4))])

    panel3 = [Translated(0,0,5.12,AxisRotated(Vector3.OX,radians(15), p)) for p in [panel, Translated(1.075,0,0,panel), Translated(2.15,0,0, panel)]]

    row = [Translated(7.5*i, 0, 0, p) for i in range(6) for p in panel3]

    panelmatrix = [Translated(3.7 if j % 2 == 0 else 7.45, 3*j, 0, geom) for j in range(6) for geom in row]

    scene = Scene([Shape(panel, id = PANELS) for panel in panelmatrix])
    return scene

    borderpole = Cylinder(0.125, 6.81, solid=False)

    # 7.58 à gacuhe
    # 7.02 à droite
    leftangle = 7.58
    rightangle = 7.02

    leftborderpoles = Group([Translated(3.4,0.96+3*i,0,Group([AxisRotated(Vector3.OY, radians(-30), AxisRotated(Vector3.OX, radians(leftangle), borderpole)),
                        AxisRotated(Vector3.OY, radians(-30), AxisRotated(Vector3.OX, radians(-rightangle), borderpole))])) for i in range(6)])

    rightborderpoles = Group([Translated(48.4,0.96+3*i,0,Group([AxisRotated(Vector3.OY, radians(30), AxisRotated(Vector3.OX, radians(leftangle), borderpole)),
                        AxisRotated(Vector3.OY, radians(30), AxisRotated(Vector3.OX, radians(-rightangle), borderpole))])) for i in range(6)])

    borderpoles = Group([leftborderpoles,rightborderpoles])

    centralpole = Cylinder(0.125/2, 5.9, solid=False)


    leftcentralpoles = Group([Translated(18.4,0.96+3*i,0,Group([ AxisRotated(Vector3.OX, radians(leftangle), centralpole),
                        AxisRotated(Vector3.OX, radians(-rightangle), centralpole)])) for i in range(6)])

    rightcentralpoles = Group([Translated(33.4,0.96+3*i,0,Group([ AxisRotated(Vector3.OX, radians(leftangle), centralpole),
                        AxisRotated(Vector3.OX, radians(-rightangle), centralpole)])) for i in range(6)])

    wireleft = Polyline([(0,0.25,5.18), (51.8,0.25,5.18)])
    wireright = Polyline([(0,0.25+1.38,5.58), (51.8,0.25+1.38,5.58)])
    wirerow = Group(wireleft,wireright)

    wires = Group([Translated(0,3*i,0,wirerow) for i in range(6)])
    scene = Scene([Shape(panelmatrix, id = PANELS), Shape(borderpoles, id = POLES), Shape(wires, id=WIRES)])
    return scene

##### TILES FOR RAY TRACING
########### COLORS FOR FLOORS
ricecol = Material("#88AA59", Color3(80,100,45))
groundcol = Material("#B68354", Color3(135,100,80))
sensorcol = Material("#94B5DA", Color3(148,181,218))
tile = QuadSet([(0,0,0),(0.5,0,0),(0.5,0.5,0),(0,0.5,0)], [list(range(4))])

def ground():

    tilecolumn = Group(tile, Group([Translated(0,0.5+0.5*i,0,tile) for i in range(30)]))
    floor = Group(tilecolumn, Group([Translated(0.5+0.5*i,0,0,tilecolumn) for i in range(99)]))

    ########### GROUND
    return  Scene([Shape(floor, groundcol, SOIL)])

def ricesensors(height = 0.7):
    ########### CANOPY HEIGHT
    floorrice = Translated(0,0,height,floor)
    rice = Shape(floorrice, ricecol, SENSORS)
    return Scene(rice)

def sensors2M(height = 2):
    ########### 2M HEIGHT : ON FIELD SENSORS
    floorsensor = Translated(0,0,2.0,floor)
    sensor = Shape(floorsensor, sensorcol, SENSORS)    
    return Scene(sensor)

if __name__ == '__main__':
    Viewer.display(generate_panels())