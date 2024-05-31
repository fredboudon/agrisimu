def lightsRepr(sun, sky, dist = 40, spheresize = 0.8, north = -90):
  import openalea.plantgl.all as pgl
  from openalea.plantgl.light import azel2vect
  from openalea.plantgl.scenegraph.colormap import PglMaterialMap
  sun_m = sun[1], sun[0], sun[2]
  directions = list(zip(*sun_m)) 
  if not sky is None:
    sky_m = sky[1], sky[0], sky[2]
    directions += list(zip(*sky_m))
  s = pgl.Scene()
  sp = pgl.Sphere(spheresize)
  cmap = PglMaterialMap(min(0,min([i for az,el,i in directions])),max([i for az,el,i in directions]))
  for az,el,i in directions:
    if i > 0:
        dir = -azel2vect(az, el, north=-north)
        s += pgl.Shape(pgl.Translated(dir*dist,sp),cmap(i))
  return s

def date(year, month, day, hour, minutes=0, seconds = 0, localisation = None):
    import pandas, datetime, pytz
    return pandas.Timestamp(datetime.datetime(year, month, day, hour, minutes, seconds), tz=pytz.timezone(localisation['timezone']))

def sunRepr(meteo, mindate = None, maxdate = None, localisation = None, dist = 40, spheresize = 0.8, north = -90):
    from alinea.astk.sun_and_sky import sun_sky_sources
    if maxdate is None:
        if mindate.hour == 0 and mindate.minute == 0:
            from copy import copy
            maxdate = date(mindate.year, mindate.month, mindate.day, 23, 59, localisation=localisation)
        else:
            maxdate = mindate
    
    currentdate = mindate
    suns = []
    for cdate, (ghi, dhi) in meteo.iterrows():
        if mindate is None or (cdate >= mindate and cdate < maxdate):
            sun, sky= sun_sky_sources(ghi = ghi, dhi = min(ghi,dhi), dates=cdate, **localisation)
            suns += sun
    return lightsRepr(suns, None, dist, spheresize, north)

if __name__ == '__main__':
   from generateplot import total_system
   from openalea.plantgl.all import Viewer
   scene, localisation, meteo = total_system()
   print(meteo)
   Viewer.display(sunRepr(meteo[0], mindate = date(2023,4,4,0, localisation=localisation), 
                                   localisation=localisation))
   #from pandas import date_range
   #
   #from alinea.astk.sun_and_sky import sun_sky_sources
   #import datetime
   #dates = date_range('2023-04-04 04:00:00+02:00','2023-04-04 21:00:00+02:00', freq=datetime.timedelta(minutes=30))
   #localisation={'latitude':47.891248, 'longitude':4.294425, 'timezone': 'Europe/Paris'}
   #sun, sky= sun_sky_sources(ghi = 1, dhi = 0, dates=dates, **localisation)
   #sc = lightsRepr(sun, sky, dist = 40, spheresize = 0.8)
   #Viewer.display(sc)