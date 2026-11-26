# Camargue 2023

This case study reconstructs irradiance beneath agrivoltaic panels at a site in Camargue using field measurements taken at nominal five-minute intervals. It compares intermittency indices across sampling intervals or treatments and relates them to crop yield.

## Data and settings

| Location | Contents |
| --- | --- |
| `data/2023_PAR_c*.txt`, `data/2023_globalradiations_c*.txt` | Tab-separated time series with local timestamps. Sensor `c1` provides the incident reference measurements (`PAR` and `Rg`). Sensors `c2` and `c3` measure irradiance beneath the panels. |
| `data/clusters.csv` | Treatment groups, positions, and field replicates |
| `data/2023_Yield_comp 1.csv` | Yield in t·ha⁻¹ by group and replicate |
| `data/Meteo_c1_2023_caribu.txt`, `data/weather.txt` | Inputs retained from the field campaign |
| `reference/` | Historical index table used for yield analysis, plus daily and seasonal animations |
| `archive/` | Exploratory versions retained for reference |

The site coordinates and time zone are defined in `localization.py` (43.734286° N, 4.570565° E, Europe/Paris). `generateplot.py` defines the 3D scene and a ground grid of 100 × 30 cells, each 0.5 m across. `meteo.py` prepares irradiance and solar position data and splits the measurements taken at 0.9 m and 2 m at noon on June 21, 2023. `data_util.py` defines the study area mask and loads the treatment groups.

## Installation

To calculate indices from existing simulations, run `python -m pip install -e .` from the repository root. For yield analysis, install the `yield` extra with `python -m pip install -e '.[yield]'`.

To run simulations, prepare weather inputs, or overlay panel outlines on maps, follow the [Conda installation instructions](../../README.md#full-conda-environment). This environment also includes the dependencies for yield analysis.

## Download and restore release data

The release includes the ground-level PlantGL cache, four irradiance ZIPs, and a manifest with the ZIP checksums and restoration paths. Download the cache to reuse PlantGL precomputations, the maps to analyze existing results, or both. Run the commands below from the repository root. [Camargue 2023 release](https://github.com/fredboudon/agrisimu/releases/tag/camargue-2023) · [Irradiance checksum manifest](https://github.com/fredboudon/agrisimu/releases/download/camargue-2023/camargue-2023-irradiance-manifest.md)

Download to a temporary directory so that verification happens before you copy any files into the case study.

```sh
release_url=https://github.com/fredboudon/agrisimu/releases/download/camargue-2023
download_dir=$(mktemp -d)
```

The examples use `shasum -a 256`. On systems with `sha256sum`, use `sha256sum FILE` to print a checksum or `sha256sum -c -` to check one from standard input.

### PlantGL cache

```sh
cache_asset=camargue-2023-pglcache0.tar.zst
cache_sha256=d519dc65e5119b1f7959bf2c22d03fde02bc3b02409298a24b26fcd8f454022b
curl -fL "$release_url/$cache_asset" -o "$download_dir/$cache_asset" &&
printf '%s  %s\n' "$cache_sha256" "$download_dir/$cache_asset" | shasum -a 256 -c -
```

After the check reports `OK`, copy the archive to the repository root.

```sh
cp -n "$download_dir/$cache_asset" .
```

When a simulation needs the cache and `examples/camargue_2023/.pglcache0/` is absent, it automatically extracts the archive there using `zstd`, included in the Conda environment. You can also keep the verified archive directly in `examples/camargue_2023/`. If a cache directory already exists, the simulation uses it. Use only trusted archives because PlantGL loads Python pickle files.

### Irradiance maps

Choose `weather2023`, `clear_sky`, `cloudy`, or `intermediate_sky` and repeat these steps for each scenario you need.

```sh
scenario=weather2023
asset="camargue-2023-irradiance-$scenario.pkl.zip"
curl -fL "$release_url/$asset" -o "$download_dir/$asset" &&
shasum -a 256 "$download_dir/$asset"
```

Compare the printed checksum with the matching row in the release manifest. Once it matches, copy the ZIP to the scenario directory under its expected filename. `cp -n` keeps any existing file at the destination. [Irradiance checksum manifest](https://github.com/fredboudon/agrisimu/releases/download/camargue-2023/camargue-2023-irradiance-manifest.md)

```sh
result_dir="examples/camargue_2023/outputs/result/$scenario"
mkdir -p "$result_dir"
cp -n "$download_dir/$asset" "$result_dir/global_irradiances.pkl.zip"
python -m examples.camargue_2023 analyze --mode resolution --results "$result_dir"
```

The analysis reads the ZIP directly. For `weather2023`, you can omit `--results`.

## Reproduce the study

Run the commands from the repository root. By default, generated files go to `examples/camargue_2023/outputs/`.

Existing plots, analyses, and sensor results are in the same directory under their original folder names, including `analysis_filtered`, `timeresolution_comparison`, and `result_sensors`.

### 1. Visualize weather data or run a simulation

```sh
python -m examples.camargue_2023 heatmap --scenario weather2023
python -m examples.camargue_2023 simulate --start 2023-06-21 --end 2023-06-22 \
    --output examples/camargue_2023/outputs/jour_test
python -m examples.camargue_2023 simulate --scenario weather2023 --jobs 2
```

The start date is inclusive and the end date is exclusive. The simulation skips timestamps where global horizontal irradiance (GHI) is zero or negative. The `weather2023` scenario uses field measurements. The `clear_sky`, `cloudy`, and `intermediate_sky` scenarios use ASTK with its default settings, an attenuation of 0.3, and an attenuation of 0.5, respectively. Use `--scenario all` to run all four scenarios.

A full simulation can produce several gigabytes of output. Each scenario writes to its own subdirectory. Rerunning in the same directory reuses previously computed TSV files or the existing aggregate. Use a different output directory if the period, geometry, or parameters change.

PlantGL stores its precomputed light results in `examples/camargue_2023/.pglcache0/` for the ground-level grid. Scenarios and workers reuse this cache across runs, even when `--output` points elsewhere. Other sensor heights use separate cache directories. After changing the 3D scene or grid geometry, remove both the cache directory and any archive for the affected height before recomputing.

### 2. Analyze intermittency

```sh
python -m examples.camargue_2023 analyze --mode resolution
python -m examples.camargue_2023 analyze --mode treatments
python -m examples.camargue_2023 analyze --mode maps
```

The `resolution` mode reproduces the original exploratory workflow. It compares results after averaging the simulated series over intervals of 5, 15, 30, and 60 minutes. The `treatments` mode compares the groups defined in `clusters.csv`, and `maps` maps the indices. By default, the analysis is restricted to the study area. Pass `--full-grid` to include all available grid points before any treatment-based selection.

By default, the analysis reads `examples/camargue_2023/outputs/result/weather2023`. It accepts existing `.pkl.zip` archives. To analyze a different scenario, pass its directory with `--results`.

```sh
python -m examples.camargue_2023 analyze --mode resolution --results examples/camargue_2023/outputs/result/cloudy
```

For the default measured campaign and study-area filter, analysis uses these directories under `outputs/`.

| Mode | Output directory |
| --- | --- |
| `resolution` | `timeresolution_comparison/` |
| `treatments` | `clusters_comparison/weather2023/` |
| `maps` | `analysis_filtered/weather2023/` |

A different simulation input or `--full-grid` uses a separate folder named after the analysis mode unless you pass `--output`.

Each analysis exports figures and a `comparison_results.csv` file. The reports reuse any stable-phase files in their output directory. Use `--output` to select a new directory when analyzing different data or changing parameters.

### 3. Analyze crop yield

```sh
python -m examples.camargue_2023 yield
python -m examples.camargue_2023 yield \
    --indices examples/camargue_2023/reference/comparison_results.csv
```

By default, yield analysis uses the per-point table in `outputs/clusters_comparison/weather2023/comparison_results.csv` when it exists and falls back to the bundled reference table. Use `--indices` to choose a different input table. The script aggregates the indices by group and replicate, saves a principal component analysis (PCA) figure, and prints the regression and Lasso selection results.

Run `python -m examples.camargue_2023.estimate_sensor` for sensor comparisons and `python -m examples.camargue_2023.sensordetection` to refine sensor positions. Sensor comparisons reuse `outputs/result_sensors/sensor_simulation.csv` when available. The fine sensor search resumes from the matching files in `outputs/optimization_result/fine/`. The [indicator guide](../../docs/analyse.md) describes the library's data conventions.
