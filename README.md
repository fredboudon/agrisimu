# AgriSimu

AgriSimu caractérise la quantité de rayonnement sous des panneaux agrivoltaïques et la succession des phases d’ombre et de lumière. Le projet fournit des méthodes d’analyse de séries temporelles et une étude de cas basée sur la campagne Camargue 2023.

## Installation

### Environnement Conda complet

Depuis la racine du dépôt, créer l’environnement décrit dans [`environnement.yaml`](environnement.yaml). Il inclut les dépendances d’analyse, de rendement et de simulation du cas Camargue.

```sh
conda env create -f environnement.yaml
conda activate agrisimu
python -m pip install --no-deps --no-build-isolation -e .
python -m examples.camargue_2023 --help
```

Le fichier utilise Python 3.14 et PlantGL 3.24.2, distribué via le canal `fredboudon/label/rc`. Cette version fournit l’API de précalcul utilisée par la simulation. [Paquets PlantGL](https://anaconda.org/channels/fredboudon/packages/openalea.plantgl/files)

### Bibliothèque d’analyse seule

Pour analyser des séries avec Python 3.10 ou ultérieur et les dépendances de `pyproject.toml` :

```sh
python -m pip install -e .
```

## Calculer un indice sur ses propres séries

L’indice spectral mesure la part de l’énergie des variations d’ombrage portée par les fréquences supérieures à un seuil. Ici, un pas de 300 secondes et une période seuil de six heures sont utilisés.

```python
import numpy as np
from agrisimu.spectral_analysis import spectral_analysis_matrix, spectral_intermittence_index

# Deux séries d’irradiance transmise, exprimées en fractions du rayonnement incident.
t = np.arange(288) * 300
transmission = np.vstack([
    0.7 + 0.2 * np.sin(2 * np.pi * t / 3600),
    0.7 + 0.2 * np.sin(2 * np.pi * t / (8 * 3600)),
])
freqs, power = spectral_analysis_matrix(1 - transmission, dt=300)
print(spectral_intermittence_index(freqs, power))
```

## Organisation du dépôt

| Emplacement | Contenu |
| --- | --- |
| [`src/agrisimu/`](src/agrisimu/) | Lecture des séries, cartes, spectres, phases stables et comparaisons |
| [`examples/camargue_2023/`](examples/camargue_2023/) | Géométrie, capteurs, mesures, commandes et résultats de référence de la campagne |
| [`docs/analyse.md`](docs/analyse.md) | Formats d’entrée, unités et définition des indicateurs |

Pour retrouver les commandes de l’étude, lancer `python -m examples.camargue_2023 --help` depuis la racine du dépôt. Le [guide Camargue 2023](examples/camargue_2023/README.md) décrit la chaîne de traitement et l’utilisation des archives existantes.

Le projet est distribué sous [licence GNU AGPL v3](LICENSE).
