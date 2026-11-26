# AgriSimu

AgriSimu characterizes solar irradiance beneath agrivoltaic panels and how shade and sunlight alternate over time. The project provides tools for analyzing time series and a case study based on the 2023 field campaign in Camargue, France.

## Installation

### Full Conda environment

From the repository root, create the environment defined in [`environnement.yaml`](environnement.yaml). It includes the dependencies for time-series analysis, crop yield analysis, and irradiance simulation.

```sh
conda env create -f environnement.yaml
conda activate agrisimu
python -m pip install --no-deps --no-build-isolation -e .
python -m examples.camargue_2023 --help
```

The environment uses Python 3.14 and PlantGL 3.24.2 from the `fredboudon/label/rc` channel. This PlantGL version provides the precomputation API used by the simulation. [PlantGL packages](https://anaconda.org/channels/fredboudon/packages/openalea.plantgl/files)

### Analysis library only

To analyze time series with Python 3.10 or later, install the library and the dependencies listed in `pyproject.toml`.

```sh
python -m pip install -e .
```

## Download the Camargue 2023 data

Download the PlantGL cache and the irradiance ZIP for the scenario you need. Verify the cache against the SHA-256 in the Camargue guide and the irradiance ZIP against the checksum manifest attached to the release. [Camargue 2023 release](https://github.com/fredboudon/agrisimu/releases/tag/camargue-2023) · [Download and verification commands](examples/camargue_2023/README.md#download-and-restore-release-data)

- **PlantGL cache.** Place the verified `camargue-2023-pglcache0.tar.zst` at the repository root. The simulation extracts it into `examples/camargue_2023/.pglcache0/` when it needs the cache and that directory is absent.
- **Irradiance maps.** Keep each verified ZIP compressed and copy it to `examples/camargue_2023/outputs/result/<scenario>/global_irradiances.pkl.zip`. Analysis uses `weather2023` by default.

## Calculate the spectral intermittency index

The spectral intermittency index measures the fraction of spectral energy in shading fluctuations above a frequency threshold. This example uses a 300-second sampling interval and a frequency threshold corresponding to a six-hour period.

```python
import numpy as np
from agrisimu.spectral_analysis import spectral_analysis_matrix, spectral_intermittence_index

# Two transmitted irradiance series expressed as fractions of incident irradiance.
t = np.arange(288) * 300
transmission = np.vstack([
    0.7 + 0.2 * np.sin(2 * np.pi * t / 3600),
    0.7 + 0.2 * np.sin(2 * np.pi * t / (8 * 3600)),
])
freqs, power = spectral_analysis_matrix(1 - transmission, dt=300)
print(spectral_intermittence_index(freqs, power))
```

## Repository layout

| Location | Contents |
| --- | --- |
| [`src/agrisimu/`](src/agrisimu/) | Time-series loading, spatial maps, spectral analysis, stable-phase detection, and group comparisons |
| [`examples/camargue_2023/`](examples/camargue_2023/) | Site geometry, sensors, measurements, commands, and reference results for the field campaign |
| [`docs/analyse.md`](docs/analyse.md) | Input formats, units, and index definitions |

Run `python -m examples.camargue_2023 --help` from the repository root to list the case study commands. The Camargue guide covers the workflow and explains how to reuse existing simulation archives. [Camargue 2023 guide](examples/camargue_2023/README.md)

AgriSimu is distributed under the [GNU AGPL v3 license](LICENSE).
