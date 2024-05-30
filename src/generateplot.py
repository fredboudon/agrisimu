from openalea.plantgl.all import *
from math import *
import pandas

PANELS, POLES, WIRES, SOIL = 1,2,3,4
SENSORS = range(10,13)
# Reflectance_Up, Transmittance_Up, Reflectance_Down, Transmittance_Down
OPTPROP = { 'PAR' :{ PANELS : (0.04,0.0,0.04,0.00), 
                     POLES  : (0.04,0.00,0.04,0.00),
                     WIRES  : (0.04,0.00,0.04,0.00),
                     SOIL   : (0.04,0.00,0.04,0.00)
                     }}

for i in SENSORS:
    OPTPROP['PAR'][i] = (0.04,0.00,0.04,0.00)


# On prend du PAR en entrée et pas du GHI

def edf_system():
    nbrows = 16
    nbcolumns = 3
    panetotlength = 7.47
    panelinterspace = 0.6
    panelsize = (2.1,(panetotlength-panelinterspace)/2)
    interrows =  (48-nbrows*panelsize[0])/(nbrows-1) if nbrows > 1 else 0 
    intercolumns = 12 # (32.1-(nbcolumns*panetotlength))/(nbcolumns-1) if nbcolumns > 1 else 0 #12 pas 32.1
    heigth = 4.58, 4.56, 4.53
    sensorwidth = 0.2 # 0.01
    sensorheight = 0.8
    sensorpos = (25.3,2.5)
    sensorpos = (18.235,1.05)

    def read_inputdata(data_file, localisation ):
        """ reader for mango meteo files """
        import pandas
        data = pandas.read_csv(data_file, delimiter = ';',
                                usecols=['date','Ray_PAR_APV_1', 'Ray_PAR_Control','Ray_PAR_Diffus_Control','Tracking_angles_alpha'], dayfirst=True)
        dates = pandas.to_datetime(data['date'], format='%d/%m/%Y %H:%M')

        data = data.rename(columns={'Ray_PAR_Control':'ghi',
                                    'Ray_PAR_Diffus_Control':'dhi',
                                    'Ray_PAR_APV_1':'sensor0'})
        # convert kW.m-2 to W.m-2
        #data['global_radiation'] *= 1000. 
        del data['date']
        index = pandas.DatetimeIndex(dates).tz_localize(localisation['timezone'])
        data = data.set_index(index)
        return data[['ghi','dhi']],data[['sensor0']],data[['Tracking_angles_alpha']]

    def genscene(orientation=0):
        poleradius = 0.05

        panel = QuadSet([(panelinterspace/2,              -panelsize[0]/2, 0),
                         (panelinterspace/2+panelsize[1], -panelsize[0]/2, 0),
                         (panelinterspace/2+panelsize[1],  panelsize[0]/2, 0),
                         (panelinterspace/2,               panelsize[0]/2, 0)], [list(range(4))])
        nb = 20
        unit = Group(#[Translated(0, 0, i*heigth/nb, Cylinder(poleradius, heigth/nb, slices = 8, solid=False)) for i in range(nb)]+
                     [Translated(0, 0, 0, AxisRotated(Vector3.OY, radians(orientation), panel) if orientation != 0 else panel), 
                      Translated(0, 0, 0, AxisRotated(Vector3.OY, radians(180+orientation),AxisRotated(Vector3.OX, radians(180),panel)) if orientation != 0 else panel)])
        
        ydecal = (panelsize[0]+interrows)*nbrows/2
        panelmatrix = [Translated(intercolumns*(j-(nbcolumns-1)/2), (panelsize[0]+interrows)*i-ydecal, height[i], unit) for i in range(nbrows) for j in range(nbcolumns)]

        scene = Scene([Shape(panel, id = PANELS) for panel in panelmatrix])

        capteur = QuadSet([(-sensorwidth/2,-sensorwidth/2,sensorheight),(sensorwidth/2,-sensorwidth/2,sensorheight),
                        (sensorwidth/2,sensorwidth/2,sensorheight),(-sensorwidth/2,sensorwidth/2,sensorheight)], [list(range(4))])
        scene.add(Shape(Translated(sensorpos[1],sensorpos[0]-ydecal,0,capteur), id = SENSORS[0]))
        return scene

    localisation={'latitude':48.3722778, 'longitude':2.8421111111111115, 'timezone': 'Europe/Paris'}

    inputdata = read_inputdata('Intercomp_measurments_renardieres(in).csv',localisation)

    return genscene, localisation, inputdata

def total_system():
    poleradius = 0.05
    panelheigth = 1.002 # 1.045*2
    panelheigthshift = 0.8
    panellong = 1.996
    interrow = 12
    sensorwidth = 0.2
    sensorheight = 0.2
    sensorpos = [0,0]
    COEF_PAR_GHI = 1.9

    def genscene(*any):


        panel = QuadSet([(0,  -panellong*11, panelheigthshift),
                         (0,  -panellong*11, panelheigthshift+panelheigth),
                         (0, panellong*11,panelheigthshift+panelheigth),
                         (0, panellong*11,panelheigth)], [list(range(4))])
        
        panelmatrix = [Translated(-interrow/2,0,0,panel),Translated(interrow/2,0,0,panel)]

        scene = Scene([Shape(panel, id = PANELS) for panel in panelmatrix])

        capteur = QuadSet([(-sensorwidth/2,-sensorwidth/2,sensorheight),(sensorwidth/2,-sensorwidth/2,sensorheight),
                        (sensorwidth/2,sensorwidth/2,sensorheight),(-sensorwidth/2,sensorwidth/2,sensorheight)], [list(range(4))])
        scene.add(Shape(Translated(sensorpos[1],sensorpos[0],0,capteur), id = SENSORS[0]))
        return scene

    localisation={'latitude':47.891248, 'longitude':4.294425, 'timezone': 'Europe/Paris'}

    def read_meteo(data_file, localisation ):
        """ reader for mango meteo files """
        import pandas
        data = pandas.read_csv(data_file, delimiter = ';',
                                usecols=['Timestamp','BNI', 'GHI','DHI','PAR','PAR_ZT'], dayfirst=True)
        dates = pandas.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M')

        data = data.rename(columns={'GHI':'ghi',
                                    'DHI':'dhi',
                                    'PAR':'sensor0'})
        data['ghi'] *=  1.9 # /4.57)
        data['dhi'] *=  1.9 #/4.57)
        # convert kW.m-2 to W.m-2
        #data['global_radiation'] *= 1000. 
        del data['Timestamp']
        index = pandas.DatetimeIndex(dates).tz_localize(localisation['timezone'])
        data = data.set_index(index)
        print(data)
        return data[['ghi','dhi']],data[['sensor0']],None

    meteo = read_meteo('donnees_PNR_modelisation_recalibration.csv',localisation)

    return genscene, localisation, meteo

def valorem_system():
    poleradius = 0.05
    panelheigth = 1.002
    panelbottom = 0.8
    paneltop = 2.6
    panellarge = 3.8
    panellong = 29.5
    interrow = 10
    sensorwidth = 0.2
    sensorheight = 0.5
    sensorpos1 = [0,-interrow/2-panellarge/2+0.5+3.3+3.1]
    sensorpos2 = [0,-interrow/2-panellarge/2+0.5+3.3]
    sensorpos3 = [0,-interrow/2-panellarge/2+0.5]
    allsensorpos = [sensorpos1,sensorpos2, sensorpos3]
    nb = 10

    def genscene(*any):


        panel = QuadSet([(-panellong/2, -panellarge/2,  panelbottom),
                         (-panellong/2, panellarge/2,   paneltop),
                         (panellong/2, panellarge/2, paneltop),
                         (panellong/2, -panellarge/2, panelbottom)], [list(range(4))])
        
        panelmatrix = [Translated(0,-interrow*(nb-1)/2+i*interrow,0,panel) for i in range(nb)]

        scene = Scene([Shape(panel, id = PANELS) for panel in panelmatrix])

        capteur = QuadSet([(-sensorwidth/2,-sensorwidth/2,sensorheight),(sensorwidth/2,-sensorwidth/2,sensorheight),
                        (sensorwidth/2,sensorwidth/2,sensorheight),(-sensorwidth/2,sensorwidth/2,sensorheight)], [list(range(4))])
        for i, sensorposi in enumerate(allsensorpos):
            scene.add(Shape(Translated(sensorposi[0],sensorposi[1],0,capteur), id = SENSORS[i]))
        return scene

    localisation={'latitude':45.3528333, 'longitude':0.4216111111111111, 'timezone': 'Europe/Paris'}

    def read_meteo(data_file, localisation ):
        """ reader for mango meteo files """
        import pandas
        data = pandas.read_csv(data_file, delimiter = ',',
                                usecols=['Date','Heure', 'Temoin_PAR','Temoin_PAR_DHI','Temoin_GHI','Interrang_GHI','Intermediaire_GHI','Panneaux_GHI'], dayfirst=True)
        dates =  data['Date']+' '+data['Heure']
        dates = pandas.to_datetime(dates) #, format='%d/%m/%Y %H:%M')
        data['dhi'] = data['Temoin_GHI'] * data['Temoin_PAR_DHI'] / data['Temoin_PAR']
        # prendre PAR
        data = data.rename(columns={'Temoin_GHI':'ghi'})
        # convert kW.m-2 to W.m-2
        #data['global_radiation'] *= 1000. 
        del data['Date']
        del data['Heure']
        index = pandas.DatetimeIndex(dates).tz_localize(localisation['timezone'])
        data = data.set_index(index)
        print(data)
        return data[['ghi','dhi']],data[['Panneaux_GHI','Intermediaire_GHI','Interrang_GHI']],None

    meteo = read_meteo('Données_rayonnement_3.csv',localisation)
    
    return genscene, localisation, meteo

if __name__ == '__main__':
    Viewer.display(edf_system()[0](20))



