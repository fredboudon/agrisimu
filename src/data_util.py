import pandas
from matplotlib import pyplot as plt
import pytz
import datetime
import numpy as np
from openalea.plantgl.light.lightmanager import to_clockwise

localisation = {'latitude':43.734286, 'longitude':4.570565, 'altitude': 0, 'timezone': 'Europe/Paris'}
# north = -90

tz = pytz.timezone(localisation['timezone'])
def date(month, day, hour):
    return pandas.Timestamp(datetime.datetime(2023, month, day, hour, 0, 0), tz=tz)


def format_values(irradiances):
    from generateplot import id2position, IDDECAL
    rows = {}
    cols = {}
    for id, irr  in irradiances.iterrows():
        if id >= IDDECAL:
            rows[id], cols[id] = id2position(id)
    irradiances = pandas.DataFrame({'row':rows, 'column':cols, 'irradiance':irradiances['irradiance'][irradiances.index>= IDDECAL]})
    irradiances = irradiances.sort_values(['column','row'])
    return irradiances

def matrix_values(df, property='irradiance'):
    import numpy as np
    res = np.zeros((max(df['row'])+1,max(df['column'])+1))
    for index, row in df.iterrows():
        res[int(row['row']),int(row['column'])] = float(row[property])
    return res

def toimage(array, fname='out.png', vmin=0, xticklabels= None, yticklabels = None, markpixel= None):
    # X in image --> correspond to second indice
    # Y in image  --> first indice
    plt.matshow(array, cmap='jet',vmin = vmin, origin='lower')
    plt.colorbar()
    if markpixel:
        plt.scatter([markpixel[1]],[markpixel[0]], c='black')
    if  xticklabels:
        nbticks = len(xticklabels)
        xrange = array.shape[0]
        plt.yticks(np.arange(0, xrange+1e-3, xrange/(nbticks-1)), labels=xticklabels)
    if yticklabels:
        nbticks = len(yticklabels)
        yrange = array.shape[1]
        plt.xticks(np.arange(0, yrange+1e-3, yrange/(nbticks-1)), labels=yticklabels)
    plt.show(block=False)
    plt.savefig(fname)
    plt.close()


def read_meteo(data_file='2023_PAR_c1.txt', localisation = localisation['timezone']):
    """ reader for mango meteo files """
    import pandas
    data = pandas.read_csv(data_file, delimiter = '\t', dayfirst=True)

    #data = data.rename(columns={'date_time':'date_time',
    #                             'par':'global_radiation'})
    # convert kW.m-2 to W.m-2
    #data['global_radiation'] *= 1000.
    index = pandas.DatetimeIndex(data['date_time']).tz_localize(localisation)
    data.drop(columns=["date_time"], inplace=True)
    data = data.set_index(index)
    return data


def get_meteo(capteur='c1'):
    ppfd = read_meteo(data_file='2023_PAR_'+capteur+'.txt')
    global_radiation = read_meteo(data_file='2023_globalradiations_'+capteur+'.txt')
    #ppfd.rename(columns={'par':'PAR'}, inplace=True)
    from openalea.astk.sky_irradiance import sky_irradiance
    sky_irr = sky_irradiance(dates=ppfd.index, ghi=global_radiation['Rg'], ppfd=ppfd['PAR'], latitude=localisation['latitude'], longitude=localisation['longitude'], altitude=localisation['altitude'])
    return sky_irr

def prepare_meteo(shadinglevel=0.2):
    c1 = get_meteo('c1')
    c2 = read_meteo('2023_globalradiations_c2.txt')
    c3 = read_meteo('2023_globalradiations_c3.txt')
    c1 = c1.loc[c1.index.isin(c2.index)]
    c1 = c1.loc[c1.index.isin(c3.index)]
    c2 = c2.loc[c2.index.isin(c1.index)]
    c3 = c3.loc[c3.index.isin(c1.index)]

    r = c2['Rg'].gt(c1['ghi'])
    c2.mask(r,c1[r]['ghi'], axis=1, inplace=True)
    r = c3['Rg'].gt(c1['ghi'])
    c3.mask(r,c1[r]['ghi'], axis=1, inplace=True)

    c1 = c1.join(c2['Rg'], how='inner')
    c1.rename(columns={'Rg':'c2'}, inplace=True)
    c1 = c1.join(c3['Rg'], how='inner')
    c1.rename(columns={'Rg':'c3'}, inplace=True)

    c1['azimuth'] = to_clockwise( c1['azimuth'])

    # compute shading ratios
    meteo = c1
    direct = meteo['ghi'] - meteo['dhi']
    meteo['c2shaded'] = meteo['ghi'] > 2 * meteo['c2']
    meteo['c3shaded'] = meteo['ghi'] > 2 *  meteo['c3']

    meteo['c2shading'] = 1- meteo['c2']/meteo['ghi']
    meteo['c3shading'] = 1- meteo['c3']/meteo['ghi']

    return meteo

def quantize_meteo(meteo, precision = 1):
    collector = {}
    for date, row in meteo.iterrows():
        azimuth, elevation = row['azimuth'], row['elevation']
        qaz = round(azimuth/precision)*precision
        qel = round(elevation/precision)*precision
        collector.setdefault((qaz, qel), [])
        collector[(qaz,qel)].append((date,row))
    propname = ['ghi','dhi','c2','c3','c2shading','c3shading','c2shaded','c3shaded']
    res = { name : [] for name in ['azimuth', 'elevation','zenith','date_time']+propname }
    for (az,el), rows in collector.items():
        res['azimuth'].append(az)
        res['elevation'].append(el)
        res['zenith'].append(90-el)
        imax = max([(row['ghi'],i) for i,(date, row) in enumerate(rows)], key=lambda x : x[0])[1]
        res['date_time'].append(rows[imax][0])
        rowmax = rows[imax][1]
        for pname in propname:
            res[pname].append(rowmax[pname])
    data = pandas.DataFrame(res)
    index = pandas.DatetimeIndex(data['date_time'])
    data.drop(columns=["date_time"], inplace=True)
    data = data.set_index(index)
    return data
        


def plot_meteo(meteo,  property='c2shading', polar = True, cmap='jet', marker='o', blocking = True, sinprojection = False):
    import matplotlib.pyplot as plt
    from openalea.plantgl.light.utils import plot_sky

    vmin, vmax = None, None
    if isinstance(property, str):
        if property == 'date_time':
            property_values = meteo.index.astype(int) / 1e9  # convert to timestamp
        elif property not in meteo.columns:
            raise ValueError('Unknown meteo property '+property)
        else:
            property_values = meteo[property]
        if property in ['c2shading','c3shading','c2shaded','c3shaded']:
            vmin, vmax = 0, 1
    else:
        property_values = property

    fig, ax = plot_sky(meteo['azimuth'], meteo['elevation'], property_values, cmap=cmap, background=None, bgresolution=1, 
             representation = 'polar' if polar else 'angle', projection ='sin' if sinprojection else 'flat', colorbarlabel = property, elevationticks = not polar, pointsize = None, edgecolors= None,
             marker = None, vmin = vmin, vmax = vmax)
    fig.set_size_inches(8, 8)
    plt.grid(False)
    plt.show(block = blocking)

def get_meteo_days(meteo):
    return np.unique(meteo.index.date)

def meteo_select_dates(meteo, selected_dates):
    if type(selected_dates) in [str,list]:
        if type(selected_dates) == str :
            if ',' in selected_dates:
                selected_dates = selected_dates.split(',')
            else:
                selected_dates = [selected_dates]
        if type(selected_dates[0]) == str :
            selected_dates = [datetime.datetime.strptime(date, "%Y/%m/%d").date() for date in selected_dates]
    mask = np.isin(meteo.index.date, selected_dates)
    selected = meteo[mask]
    return selected

def meteo_subsampling(meteo, nbdates = 50, seed = 0):
    dates = get_meteo_days(meteo)
    if nbdates > len(dates) : return meteo
    np.random.seed(seed)
    selected_dates = np.random.choice(dates, size=nbdates, replace=False)
    return meteo_select_dates(meteo, selected_dates)


meteo = prepare_meteo()
clearsky = meteo[meteo['ghi'] *0.5 > meteo['dhi']]
clearsky1 = quantize_meteo(clearsky,1)


def plot_sensor_meteo(saving = True, blocking = False, polar = True, output = 'dataview'):
    import matplotlib.pyplot as plt
    import os
    suffix = ''
    if not os.path.exists(output):
        os.mkdir(output)
    for sensor in ['c2','c3']:
        plot_meteo(meteo,'ghi', blocking=blocking, polar=polar)
        if saving:
            plt.savefig(os.path.join(output,sensor+'_A_meteo'+suffix+'.png'))
            plt.close()
        
        plot_meteo(meteo, sensor, blocking=blocking, polar=polar)
        if saving:
            plt.savefig(os.path.join(output,sensor+'_B_sensor'+suffix+'.png'))
            plt.close()

        plot_meteo(meteo,sensor+'shading', blocking=blocking, polar=polar)
        if saving:
            plt.savefig(os.path.join(output,sensor+'_C_shading'+suffix+'.png'))
            plt.close()

        plot_meteo(clearsky,sensor+'shading', blocking=blocking, polar=polar)
        if saving:
            plt.savefig(os.path.join(output,sensor+'_D_clearsky_shading'+suffix+'.png'))
            plt.close()

        plot_meteo(clearsky,sensor+'shaded', blocking=blocking, polar=polar, cmap='binary')
        if saving:
            plt.savefig(os.path.join(output,sensor+'_E_clearsky_shaded'+suffix+'.png'))
            plt.close()

        plot_meteo(clearsky1,sensor+'shaded', blocking=blocking, polar=polar, cmap='binary')
        if saving:
            plt.savefig(os.path.join(output,sensor+'_F_quantized_clearsky_shaded'+suffix+'.png'))
            plt.close()

if __name__ == '__main__':
    #print(len(clearsky))
    #simplerclearsky = quantize_meteo(clearsky,1)
    #print(len(simplerclearsky))
    #plot_meteo(clearsky1, "c3shading",cmap='jet')
    #m = meteo_subsampling(meteo,1, seed=1)
    #meteo[['ghi','dhi','c2','c3']].plot()
    plot_sensor_meteo()
