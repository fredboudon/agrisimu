from openalea.plantgl.all import *
from math import *

PANELS, POLES, WIRES, SOIL, SENSORS = 1,2,3,4,5
# Reflectance_Up, Transmittance_Up, Reflectance_Down, Transmittance_Down
OPTPROP = { 'PAR' :{ PANELS : (0.04,0.0,0.04,0.00), 
                     POLES  : (0.04,0.00,0.04,0.00),
                     WIRES  : (0.04,0.00,0.04,0.00),
                     SOIL   : (0.04,0.00,0.04,0.00)
                     }}

nbpanels = 9
ricerowwidth = 3
panelinterval = 2

def generate_plots(heigth = 1.5, orientation=20, shift = 1.2):
    width = 1.5
    poleradius = 0.02
    length = 3

    heightdiff = width*sin(radians(orientation))
    widthdiff = width*cos(radians(orientation))
    panel = QuadSet([(0,0,0),(length,0,0),(length,width,0),(0,width,0)], [list(range(4))])
    
    
    orientedpanel = Translated(0,width-widthdiff+shift,heigth,AxisRotated(Vector3.OX,radians(orientation),panel)) #original orientedpanel = Translated(0,width-widthdiff,heigth,AxisRotated(Vector3.OX,radians(orientation),panel))
    orientedpanel2 = Translated(0,4.5-shift,heigth+heightdiff,AxisRotated(Vector3.OX,-radians(orientation),panel))# original orientedpanel2 = Translated(0,4.5,heigth+heightdiff,AxisRotated(Vector3.OX,-radians(orientation),panel))

    panelmatrix = [Translated(5*i, 0, 0, ipanel) for i in range(nbpanels) for ipanel in [orientedpanel,orientedpanel2]]

    scene = Scene([Shape(panel, id = PANELS) for panel in panelmatrix])

    #outerpole = Cylinder(poleradius, heigth, solid=False)
    #innerpole = Cylinder(poleradius, heigth+heightdiff, solid=False)

    #outerpoles = [Translated(r,c,0,outerpole) for c in [0,2*width+ricerowwidth] 
     #          for r in [i*(length+panelinterval) for i in range(nbpanels)]+[i*(length+panelinterval)+length for i in range(nbpanels)]]

    #innerpoles = [Translated(r,c,0,innerpole) for c in [width,width+ricerowwidth] 
     #          for r in [i*(length+panelinterval) for i in range(nbpanels)]+[i*(length+panelinterval)+length for i in range(nbpanels)]]

    #scene += Scene([Shape(pole, id = POLES) for pole in outerpoles+innerpoles])
    return scene

##### TILES FOR RAY TRACING
########### COLORS FOR FLOORS
ricecol = Material("#88AA59", Color3(80,100,45))
groundcol = Material("#B68354", Color3(135,100,80))
sensorcol = Material("#94B5DA", Color3(148,181,218))

tilewidth = 0.2
tilelength = 1/3.
tile = QuadSet([(0,0,0),(tilelength,0,0),(tilelength,tilewidth,0),(0,tilewidth,0)], [list(range(4))])

def groundids():
    return [(col, rank)  for rank in range(15) for col in  [100*i+c for c in range(9) for i in range(nbpanels)]]

def ground():
    floor = [Translated(tilelength*(col%100) + 5*(col//100),1.5+tilewidth*rank,0,tile) for col, rank in groundids()]
    ########### GROUND
    return  Scene([Shape(square, groundcol, SOIL) for square in floor])



if __name__ == '__main__':
    Viewer.display(generate_plots())



