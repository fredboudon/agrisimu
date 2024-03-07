from openalea.plantgl.all import *
from math import *

PANELS, POLES, WIRES, SOIL, SENSORS = 1,2,3,4,5
# Reflectance_Up, Transmittance_Up, Reflectance_Down, Transmittance_Down
OPTPROP = { 'PAR' :{ PANELS : (0.04,0.0,0.04,0.00), 
                     POLES  : (0.04,0.00,0.04,0.00),
                     WIRES  : (0.04,0.00,0.04,0.00),
                     SOIL   : (0.04,0.00,0.04,0.00)
                     }}

nbpanels = 10 # 200
panelwidth = 1.7
panellength = 50 # 130

def generate_plots(orientation=22.5):
    heigth = 1.5

    panel = QuadSet([(0,0,0),(panellength,0,0),(panellength,panelwidth,0),(0,panelwidth,0)], [list(range(4))])
    
    
    orientedpanel = Translated(0,0,heigth,AxisRotated(Vector3.OX,radians(orientation),panel)) #original orientedpanel = Translated(0,width-widthdiff,heigth,AxisRotated(Vector3.OX,radians(orientation),panel))

    panelmatrix = [Translated(0, 3.2*i, 0, orientedpanel) for i in range(nbpanels) ]

    scene = Scene([Shape(panel, id = PANELS) for panel in panelmatrix])

    return scene

##### TILES FOR RAY TRACING
########### COLORS FOR FLOORS
ricecol = Material("#88AA59", Color3(80,100,45))
groundcol = Material("#B68354", Color3(135,100,80))
sensorcol = Material("#94B5DA", Color3(148,181,218))

tilewidth = 0.5
tile = QuadSet([(0,0,0),(tilewidth,0,0),(tilewidth,tilewidth,0),(0,tilewidth,0)], [list(range(4))])

def groundids():
    return [(col, rank)  for rank in range(int((3.2*nbpanels)//tilewidth)) for col in  range(int(panellength//tilewidth)) ]

def ground(heigth = 0):
    floor = [Translated(tilewidth*col,tilewidth*rank,heigth,tile) for col, rank in groundids()]
    ########### GROUND
    return  Scene([Shape(square, groundcol, SOIL) for square in floor])



if __name__ == '__main__':
    Viewer.display(generate_plots()+ground())



