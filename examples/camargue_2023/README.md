# Camargue 2023

Le cas d’usage reconstitue l’irradiance sous les panneaux d’une parcelle de Camargue à partir d’une campagne de mesures à pas nominal de cinq minutes. Il permet de comparer les indices d’intermittence selon la résolution temporelle ou le traitement, puis de les relier au rendement.

## Données et paramètres

| Emplacement | Contenu |
| --- | --- |
| `data/2023_PAR_c*.txt`, `data/2023_globalradiations_c*.txt` | Séries tabulées horodatées en heure locale. `c1` fournit la référence incidente (`PAR` et `Rg`), `c2` et `c3` les mesures sous panneaux. |
| `data/clusters.csv` | Groupes de traitement, positions et répétitions de la parcelle |
| `data/2023_Yield_comp 1.csv` | Rendement en t·ha⁻¹ par groupe et répétition |
| `data/Meteo_c1_2023_caribu.txt`, `data/weather.txt` | Entrées conservées de la campagne |
| `reference/` | Table d’indices historique utilisée pour le rendement et animations journalière et saisonnière |
| `archive/` | Versions exploratoires conservées pour consultation |

La localisation est définie dans `localization.py` (43,734286° N, 4,570565° E, fuseau Europe/Paris). `generateplot.py` contient la scène 3D et la grille au sol de 100 × 30 cellules de 0,5 m. `meteo.py` prépare le rayonnement et la position du soleil et sépare les périodes de mesure à 0,9 m et à 2 m au 21 juin 2023 à midi. `data_util.py` décrit le masque de la zone étudiée et la lecture des groupes de traitement.

## Installation

Pour calculer les indices à partir de simulations existantes, installer le projet depuis la racine avec `python -m pip install -e .`. Pour l’analyse de rendement, installer aussi l’extension `yield` avec `python -m pip install -e '.[yield]'`.

Pour simuler, préparer la météo ou superposer les panneaux sur les cartes, suivre les [instructions Conda du README principal](../../README.md#environnement-conda-complet). Cet environnement inclut aussi les dépendances de l’analyse du rendement.

## Rejouer l’étude

Lancer les commandes depuis la racine du dépôt. Par défaut les fichiers produits vont dans `examples/camargue_2023/outputs/`.

### 1. Visualiser la météo ou simuler

```sh
python -m examples.camargue_2023 heatmap --scenario weather2023
python -m examples.camargue_2023 simulate --start 2023-06-21 --end 2023-06-22 \
    --output examples/camargue_2023/outputs/jour_test
python -m examples.camargue_2023 simulate --scenario weather2023 --jobs 2
```

La date de début est incluse et la date de fin est exclue. La simulation saute les dates dont le GHI est nul ou négatif. Le scénario `weather2023` part des mesures. `clear_sky`, `cloudy` et `intermediate_sky` utilisent ASTK avec ses paramètres par défaut, une atténuation de 0,3 et une atténuation de 0,5, respectivement. `--scenario all` lance les quatre scénarios.

Une simulation complète peut produire plusieurs gigaoctets. Chaque scénario écrit dans son propre sous-répertoire. Une relance au même endroit reprend les TSV déjà calculés ou réutilise l’agrégat existant. Utiliser un autre répertoire de sortie si la période, la géométrie ou les paramètres changent.

### 2. Analyser l’intermittence

```sh
python -m examples.camargue_2023 analyze --mode resolution
python -m examples.camargue_2023 analyze --mode treatments
python -m examples.camargue_2023 analyze --mode maps
```

`resolution` correspond au point d’entrée exploratoire d’origine. Il compare les moyennes calculées à pas de 5, 15, 30 et 60 minutes à partir des séries simulées. `treatments` compare les groupes du fichier `clusters.csv` et `maps` cartographie les indicateurs. L’analyse se concentre par défaut sur la zone d’étude. `--full-grid` inclut tous les points disponibles avant la sélection éventuelle par traitement.

Pour exploiter une campagne simulée avant la restructuration, passer le dossier existant. Ce mode lit aussi les archives `.pkl.zip`.

```sh
python -m examples.camargue_2023 analyze --mode resolution --results src/result/weather2023
```

Chaque analyse exporte des figures et un fichier `comparison_results.csv`. Les rapports réutilisent les fichiers de phases stables présents dans leur dossier de sortie. Choisir un nouveau dossier via `--output` pour analyser d’autres données ou paramètres.

### 3. Analyser le rendement

```sh
python -m examples.camargue_2023 yield
python -m examples.camargue_2023 yield \
    --indices examples/camargue_2023/outputs/treatments/comparison_results.csv
```

La première commande utilise la table de référence fournie. La seconde utilise les indices par point issus de l’analyse par traitement. Le script agrège les indices par groupe et répétition, enregistre la figure d’ACP et imprime les résultats des régressions et de la sélection Lasso.

Les modules `estimate_sensor` et `sensordetection` restent exécutables avec `python -m examples.camargue_2023.estimate_sensor` et `python -m examples.camargue_2023.sensordetection` pour les comparaisons et l’ajustement de position des capteurs. Le [guide des indicateurs](../../docs/analyse.md) décrit les conventions de données de la bibliothèque.
