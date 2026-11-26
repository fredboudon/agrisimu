"""Simulation PlantGL de l’irradiance sur la grille de Camargue."""

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

from openalea.plantgl.all import LightEstimator, eShapeBased, eTriangleProjection

from agrisimu.data_util import compress_irradiances

from .data_util import format_values
from .generateplot import agristructure, sensorgeometry
from .localization import localisation
from .paths import OUTPUT_DIR

DEFAULT_RESULT = OUTPUT_DIR / "result" / "weather2023"


def _daylight_weather(meteo, mindate, maxdate):
    if mindate is not None:
        meteo = meteo.loc[meteo.index >= mindate]
    if maxdate is not None:
        meteo = meteo.loc[meteo.index < maxdate]
    return meteo.loc[meteo["ghi"] > 0]


def _has_archive(outdir):
    return any((outdir / name).is_file() for name in ("global_irradiances.pkl", "global_irradiances.pkl.zip"))


def process_light(meteo, mindate=None, maxdate=None, outdir=DEFAULT_RESULT, jobname=None,
                  sensorheight=0, multithreaded=True):
    """Simule un lot de dates et retourne les tables ou les chemins de leurs TSV."""
    weather = _daylight_weather(meteo, mindate, maxdate)
    if weather.empty:
        return []

    outdir = Path(outdir) if outdir is not None else None
    height = str(sensorheight).replace(".", "_")
    if outdir is not None:
        if _has_archive(outdir):
            print(f"Archive présente dans {outdir}")
            return []
        outdir.mkdir(parents=True, exist_ok=True)
        cache = outdir / f".pglcache_{height}_{jobname}"
    else:
        cache = None

    structure = agristructure()
    sensors = sensorgeometry(sensorheight)
    estimator = LightEstimator(sensors + structure)
    estimator.localize(name="Camargue", **localisation)
    estimator.set_method(
        method=eTriangleProjection, primitive=eShapeBased,
        occludedOnly={shape.id for shape in sensors},
        occludingOnly={shape.id for shape in structure},
        multithreaded=multithreaded,
    )
    estimator.always_precompute()
    if cache is not None:
        estimator.load_precomputation(str(cache))

    results = []
    for timestamp, row in weather.iterrows():
        if outdir is not None:
            path = outdir / f"simulation_{height}_{timestamp:%Y-%m-%d-%H-%M}.csv"
            if path.is_file():
                results.append((timestamp, str(path)))
                continue
        estimator.clear_lights()
        estimator.add_light(
            f"sun_{timestamp:%Y%m%d_%H%M}", row["elevation"], row["azimuth"],
            row["ghi"] - row["dhi"], horizontal=True, date=timestamp, type="SUN",
        )
        estimator.add_sky(irradiance=row["dhi"])
        result = format_values(estimator())
        result["TrIrradiance"] = result["irradiance"] / row["ghi"]
        if outdir is None:
            results.append((timestamp, result))
        else:
            partial = path.with_suffix(".tmp")
            result.to_csv(partial, sep="\t")
            partial.replace(path)
            estimator.dump_precomputation(str(cache))
            results.append((timestamp, str(path)))
        print(f"Simulation terminée pour {timestamp}")

    if outdir is not None and jobname is None:
        compress_irradiances(outdir)
    return results


def _process_light(args):
    weather, outdir, job, height = args
    return process_light(weather, outdir=outdir, jobname=job, sensorheight=height, multithreaded=False)


def mt_process_light(meteo, mindate=None, maxdate=None, nbjobs=None, *, outdir=DEFAULT_RESULT,
                     sensorheight=0):
    """Répartit les dates entre processus et consolide les fichiers du scénario."""
    outdir = Path(outdir)
    if _has_archive(outdir):
        print(f"Archive présente dans {outdir}")
        return []

    weather = _daylight_weather(meteo, mindate, maxdate)
    if weather.empty:
        raise ValueError("Aucune mesure avec GHI positif pour la période demandée.")

    workers = min(nbjobs if nbjobs is not None else (os.cpu_count() or 1), len(weather))
    if workers < 1:
        raise ValueError("Le nombre de processus doit être positif.")
    if workers == 1:
        results = [process_light(weather, outdir=outdir, jobname=0, sensorheight=sensorheight)]
    else:
        batch_size = (len(weather) + workers - 1) // workers
        batches = [
            (weather.iloc[offset:offset + batch_size], outdir, job, sensorheight)
            for job, offset in enumerate(range(0, len(weather), batch_size))
        ]
        with ProcessPoolExecutor(max_workers=len(batches)) as pool:
            results = list(pool.map(_process_light, batches))
    compress_irradiances(outdir)
    return results


if __name__ == "__main__":
    from .cli import main

    main(["simulate", "--scenario", "all"])
