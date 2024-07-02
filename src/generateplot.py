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

    return scene

plotwidth = 1.35
plotlength = 1.65

def generate_plots2(nb = 5, panelheight = 0.5, height = 0.5):

    spacing = plotwidth/nb
    panel = QuadSet([(0,0,height),(plotlength,0,height),(plotlength,0,height+panelheight),(0,0,height+panelheight)], [list(range(4))])
    panelmatrix = [Translated(0,spacing*i,0,panel) for i in range(nb+1)]
    scene = Scene([Shape(panel, id = PANELS) for panel in panelmatrix])
    print('Surface:',surface(scene))
    borderpanel = QuadSet([(0,0,height),(0,plotwidth,height),(0,plotwidth,height+panelheight),(0,0,height+panelheight)], [list(range(4))])
    scene += Scene([Shape(borderpanel, id = PANELS),Shape(Translated(plotlength,0,0,borderpanel), id = PANELS)])
    return scene



##### TILES FOR RAY TRACING
########### COLORS FOR FLOORS
ricecol = Material("#88AA59", Color3(80,100,45))
groundcol = Material("#B68354", Color3(135,100,80))
sensorcol = Material("#94B5DA", Color3(148,181,218))

tilewidth = 0.075
tilelength = tilewidth
decal = 0.6
tile = QuadSet([(0,0,0),(tilelength,0,0),(tilelength,tilewidth,0),(0,tilewidth,0)], [list(range(4))])

def groundids():
    return [(col, rank)  for rank in range(ceil((plotlength+2*decal)/tilelength)) for col in range(ceil((plotwidth+2*decal)/tilewidth))]

def ground():
    floor = [Translated(tilelength*rank - decal,tilewidth*col-decal,0,tile) for col, rank in groundids()]
    ########### GROUND
    return  Scene([Shape(square, groundcol, SOIL) for square in floor])



if __name__ == '__main__':
    Viewer.display(generate_plots2()+ground())



