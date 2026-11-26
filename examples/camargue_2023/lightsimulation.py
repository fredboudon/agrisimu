"""PlantGL irradiance simulation on the Camargue grid."""

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from openalea.plantgl.all import LightEstimator, eShapeBased, eTriangleProjection

from agrisimu.data_util import compress_irradiances

from .cache_archive import restore_cache
from .data_util import format_values
from .generateplot import agristructure, sensorgeometry
from .localization import localisation
from .paths import CASE_DIR, MEASURED_RESULTS

DEFAULT_RESULT = MEASURED_RESULTS


def _daylight_weather(meteo, mindate, maxdate):
    if mindate is not None:
        meteo = meteo.loc[meteo.index >= mindate]
    if maxdate is not None:
        meteo = meteo.loc[meteo.index < maxdate]
    return meteo.loc[meteo["ghi"] > 0]


def _has_archive(outdir):
    return any((outdir / name).is_file() for name in ("global_irradiances.pkl", "global_irradiances.pkl.zip"))


def _save_precomputation(estimator, cache):
    """Publish complete pickle files so workers can share the cache."""
    cache.mkdir(exist_ok=True)
    for light_id, value in estimator.precomputed_lights.items():
        if value is None or isinstance(value, str):
            continue
        path = cache / f"{light_id}.pkl"
        if path.exists():
            continue
        with NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
        try:
            value.to_pickle(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def process_light(meteo, mindate=None, maxdate=None, outdir=DEFAULT_RESULT, jobname=None,
                  sensorheight=0, multithreaded=True):
    """Simulate a batch of timestamps and return tables or paths to their TSV files."""
    weather = _daylight_weather(meteo, mindate, maxdate)
    if weather.empty:
        return []

    outdir = Path(outdir) if outdir is not None else None
    height = str(sensorheight).replace(".", "_")
    if outdir is not None:
        if _has_archive(outdir):
            print(f"Reusing the existing simulation archive in {outdir}")
            return []
        outdir.mkdir(parents=True, exist_ok=True)
    cache = CASE_DIR / f".pglcache{height}"

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
    restore_cache(cache)
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
            _save_precomputation(estimator, cache)
            results.append((timestamp, str(path)))
        print(f"Simulation completed for {timestamp}")

    if outdir is not None and jobname is None:
        compress_irradiances(outdir)
    return results


def _process_light(args):
    weather, outdir, job, height = args
    return process_light(weather, outdir=outdir, jobname=job, sensorheight=height, multithreaded=False)


def mt_process_light(meteo, mindate=None, maxdate=None, nbjobs=None, *, outdir=DEFAULT_RESULT,
                     sensorheight=0):
    """Distribute timestamps among worker processes and combine the scenario outputs."""
    outdir = Path(outdir)
    if _has_archive(outdir):
        print(f"Reusing the existing simulation archive in {outdir}")
        return []

    weather = _daylight_weather(meteo, mindate, maxdate)
    if weather.empty:
        raise ValueError("No measurements with positive GHI in the requested period.")

    workers = min(nbjobs if nbjobs is not None else (os.cpu_count() or 1), len(weather))
    if workers < 1:
        raise ValueError("The number of worker processes must be positive.")
    if workers == 1:
        results = [process_light(weather, outdir=outdir, jobname=0, sensorheight=sensorheight)]
    else:
        height = str(sensorheight).replace(".", "_")
        restore_cache(CASE_DIR / f".pglcache{height}")
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
