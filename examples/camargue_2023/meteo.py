"""Weather data preparation for the Camargue field campaign."""

import pandas
import pandas as pd
from dataclasses import dataclass

from .localization import localisation as SITE
from .paths import DATA_DIR


date2m = pd.Timestamp('2023-06-21 12:00', tz=SITE['timezone'])


def read_meteo(data_file=DATA_DIR / '2023_PAR_c1.txt', localisation=SITE['timezone']):
    """ reader for mango meteo files """
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
    ppfd = read_meteo(data_file=DATA_DIR / f'2023_PAR_{capteur}.txt')
    global_radiation = read_meteo(data_file=DATA_DIR / f'2023_globalradiations_{capteur}.txt')
    #ppfd.rename(columns={'par':'PAR'}, inplace=True)
    from openalea.astk.sky_irradiance import sky_irradiance
    sky_irr = sky_irradiance(dates=ppfd.index, ghi=global_radiation['Rg'], ppfd=ppfd['PAR'], latitude=SITE['latitude'], longitude=SITE['longitude'], altitude=SITE['altitude'])
    return sky_irr

def prepare_meteo(shadinglevel=0.5):
    c1 = get_meteo('c1')
    c2 = read_meteo(data_file=DATA_DIR / '2023_globalradiations_c2.txt')
    c3 = read_meteo(data_file=DATA_DIR / '2023_globalradiations_c3.txt')
    c1 = c1.loc[c1.index.isin(c2.index)]
    c1 = c1.loc[c1.index.isin(c3.index)]
    c2 = c2.loc[c2.index.isin(c1.index)]
    c3 = c3.loc[c3.index.isin(c1.index)]

    r = c2['Rg'].gt(c1['ghi'])
    c2.mask(r,c1[r]['ghi'], axis=1, inplace=True)
    r = c3['Rg'].gt(c1['ghi'])
    c3.mask(r,c1[r]['ghi'], axis=1, inplace=True)

    meteo = c1
    meteo = meteo.join(c2['Rg'], how='inner')
    meteo.rename(columns={'Rg':'c2'}, inplace=True)
    meteo = meteo.join(c3['Rg'], how='inner')
    meteo.rename(columns={'Rg':'c3'}, inplace=True)

    #meteo['azimuth'] = to_clockwise( meteo['azimuth'])

    # compute shading ratios
    meteo['c2shading'] = 1- meteo['c2']/meteo['ghi']
    meteo['c3shading'] = 1- meteo['c3']/meteo['ghi']

    # Estimate shading presence with a threshold on shading ratio
    meteo['c2shaded'] = meteo['c2shading'] > shadinglevel
    meteo['c3shaded'] = meteo['c3shading'] > shadinglevel

    # Drop the incomplete first and last days
    meteo = meteo.loc[
    (meteo.index.date != meteo.index.date[0]) &
    (meteo.index.date != meteo.index.date[-1])]

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

@dataclass
class Weather:
    """Measured weather data, clear-sky subsets, and rounded solar positions."""

    meteo: pd.DataFrame
    meteo90: pd.DataFrame
    meteo200: pd.DataFrame
    clearsky: pd.DataFrame
    clearsky90: pd.DataFrame
    clearsky200: pd.DataFrame
    clearskyQ: pd.DataFrame
    clearskyQ90: pd.DataFrame
    clearskyQ200: pd.DataFrame


def setup_meteo():
    """Load campaign data and split it by sensor height period."""
    meteo = prepare_meteo()
    before = meteo.index < date2m
    clear = meteo['ghi'] * 0.5 > meteo['dhi']
    periods = (meteo, meteo[before], meteo[~before])
    clear_periods = (meteo[clear], meteo[clear & before], meteo[clear & ~before])
    quantized = tuple(quantize_meteo(period) for period in clear_periods)
    return Weather(*periods, *clear_periods, *quantized)


def generate_meteo(dates, attenuation=None):
    """Generate an irradiance scenario for the campaign timestamps."""
    from openalea.astk.sky_irradiance import sky_irradiance

    return sky_irradiance(
        dates=dates, attenuation=attenuation,
        latitude=SITE['latitude'], longitude=SITE['longitude'], altitude=SITE['altitude'],
    )
