from openalea.plantgl.all import *
from math import *

PANELS, POLES, WIRES, SOIL, SENSORS = 1,2,3,4,5
# Reflectance_Up, Transmittance_Up, Reflectance_Down, Transmittance_Down
OPTPROP = { 'PAR' :{ PANELS : (0.04,0.0,0.04,0.00), 
                     POLES  : (0.04,0.00,0.04,0.00),
                     WIRES  : (0.04,0.00,0.04,0.00),
                     SOIL   : (0.04,0.00,0.04,0.00)
                     }}

l,w  = 7.326, 2.094
interpanel = 8.000
interrow = 5.500
heigth = 5.755

def generate_plots(angle = 0):
    print('Angle:', angle)
    panel = AxisRotated((0,1,0),radians(angle),QuadSet([(-w/2,-l/2,0),(-w/2,l/2,0),(w/2,l/2,0),(w/2,-l/2,0)], [list(range(4))]))


    row3 = [Translated(0, interpanel*(i-3.5),  0, panel) for i in range(3)] # 2 
    row8 = [Translated( 0, interpanel*(i-3.5), 0, panel) for i in range(8)] # 9

    panelmatrix = [Translated(interrow*(j-5.5), 5*interpanel, heigth, geom) for j in range(2) for geom in row3]+[Translated(interrow*(j-5.5), 0, heigth, geom) for j in range(2,11) for geom in row8]

    scene = Scene([Shape(panel, id = PANELS) for panel in panelmatrix])

    #borderpole = Cylinder(0.125, 6.81, solid=False)

    # 7.58 à gauche
    # 7.02 à droite
    #leftangle = 7.58
    #rightangle = 7.02


    #leftborderpoles = [Translated(3.4,0.96+3*i,0,p) for p in [AxisRotated(Vector3.OY, radians(-30), AxisRotated(Vector3.OX, radians(leftangle), borderpole)), AxisRotated(Vector3.OY, radians(-30), AxisRotated(Vector3.OX, radians(-rightangle), borderpole))] for i in range(5)]

    
    #rightborderpoles = [Translated(48.4,0.96+3*i,0,p) for p in [AxisRotated(Vector3.OY, radians(30), AxisRotated(Vector3.OX, radians(leftangle), borderpole)), AxisRotated(Vector3.OY, radians(30), AxisRotated(Vector3.OX, radians(-rightangle), borderpole))] for i in range(5)]

    #borderpoles = leftborderpoles + rightborderpoles


    #centralpole = Cylinder(0.125/2, 5.9, solid=False)

    
    #leftcentralpoles = [Translated(18.4,0.96+3*i,0,p) for p in [AxisRotated(Vector3.OX, radians(leftangle), centralpole),
    #                    AxisRotated(Vector3.OX, radians(-rightangle), centralpole)] for i in range(5)]

    #rightcentralpoles = [Translated(33.4,0.96+3*i,0,p) for p in [AxisRotated(Vector3.OX, radians(leftangle), centralpole),
    #                    AxisRotated(Vector3.OX, radians(-rightangle), centralpole)] for i in range(5)]

    #scene += Scene([Shape(pole, id = POLES) for pole in borderpoles+leftcentralpoles+rightcentralpoles])
    return scene

def read_meteo(data_file='AOS-872 - PRATX RIVESALTES 1-23_07_2023-23_07_2024_&_ANGLES.csv', localisation = None):
    """ reader for meteo files """
    import pandas
    import locale
    locale.setlocale(locale.LC_NUMERIC, '')
    from locale import atof
    cols = ['date','Solargis/diffuse_horizontal_irradiation','Solargis/global_horizontal_irradiation','Angle']
    data = pandas.read_csv(data_file, delimiter = ';', usecols=cols)
    print(data)
    dates = pandas.to_datetime(data['date'], dayfirst=True) #, format='%d/%m/%Y %H:%M')

    data = data.rename(columns={'Solargis/diffuse_horizontal_irradiation':'pardiffus',
                                'Solargis/global_horizontal_irradiation':'par',
                                })
    del data['date']
    index = pandas.DatetimeIndex(dates, tz=localisation['timezone'], ambiguous=False)
    #if localisation:
    #    index = index.tz_localize(localisation['timezone'])
    data = data.set_index(index)
    print(data[['Angle']].dtypes)
    return data[['par','pardiffus']], data[['Angle']]


##### TILES FOR RAY TRACING
########### COLORS FOR FLOORS
ricecol = Material("#88AA59", Color3(80,100,45))
groundcol = Material("#B68354", Color3(135,100,80))
sensorcol = Material("#94B5DA", Color3(148,181,218))
sensorsize =(0.5,0.5)
tile = QuadSet([(0,0,0),(sensorsize[0],0,0),(sensorsize[0],sensorsize[1],0),(0,sensorsize[1],0)], [list(range(4))])

def groundids():
    return [(col, rank)  for rank in range(round(interpanel*8/sensorsize[1])) for col in range(round(interrow*11/sensorsize[0]))]

def ground(height = 0):
    floor = [Translated(0.5*col-interrow*6,0.5*rank-interpanel*4,height,tile) for col, rank in groundids()]
    ########### GROUND
    return  Scene([Shape(square, groundcol, SOIL) for square in floor])

if __name__ == '__main__':
    Viewer.display(generate_plots()+ground())