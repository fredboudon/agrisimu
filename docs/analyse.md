# Analyser des séries d’irradiance

## Entrées

Les rapports de `agrisimu.irradiance_processing` prennent des tableaux pandas indexés par date, regroupés dans un dictionnaire `{(colonne, ligne): DataFrame}`. Les coordonnées sont des indices de grille entiers commençant à zéro. Chaque point d’un groupe doit avoir les mêmes dates, dans l’ordre chronologique. Les deux colonnes nécessaires sont les suivantes.

| Colonne | Définition |
| --- | --- |
| `irradiance` | Irradiance reçue au point simulé, en W·m⁻² |
| `TrIrradiance` | Rapport entre l’irradiance reçue et le rayonnement global horizontal incident (GHI) au même instant |

L’ombrage utilisé dans les calculs est `S = 1 - TrIrradiance`. Si les données proviennent d’un fichier tabulaire externe, convertir les dates en `DatetimeIndex` et regrouper la table par `column` et `row`. `retrieve_irradiance_per_plot` réalise ce regroupement pour une table dont les dates sont dans `date_time`.

`get_irradiances_per_plot(path)` lit un répertoire de simulation. Il accepte les TSV `simulation_<hauteur>_<AAAA-MM-JJ-HH-MM>.csv` et les agrégats historiques `global_irradiances.pkl` ou `global_irradiances.pkl.zip`. La hauteur est en mètres, avec un soulignement comme séparateur décimal. Les dates des TSV sont des heures locales sans fuseau dans le nom du fichier. Les fichiers pickle doivent provenir d’une source de confiance.

## Indicateurs

| Colonne | Calcul |
| --- | --- |
| `mean_irradiance` | Moyenne des irradiances simulées en W·m⁻² |
| `mean_shading` | Moyenne temporelle de l’ombrage |
| `Iq` | Moyenne de la valeur absolue des différences successives d’ombrage, ramenée à une heure |
| `Isp` | Part de l’énergie spectrale au-dessus de la fréquence correspondant à six heures. Une série constante a une énergie nulle et l’indice est indéfini. |
| `If` | Durée cumulée des phases stables divisée par la durée du premier au dernier relevé |
| `It` | Durée moyenne des phases stables en minutes |
| `Is` | Moyenne des niveaux moyens d’ombrage des phases stables |

La détection des phases stables s’appuie sur l’écart-type glissant d’un signal lissé par un filtre gaussien et traite chaque jour séparément. La durée d’une phase va de son premier à son dernier échantillon. Les indices `It` et `Is` sont absents quand aucune phase stable n’est détectée. Les rapports calculent l’ACP sur les points dont tous les indices utilisés sont renseignés.

Les rapports déduisent le pas nominal des deux premières dates. La FFT traite les échantillons successifs comme équidistants, y compris lorsqu’une série de campagne omet les nuits. Les différences utilisées par `Iq` incluent également les paires qui encadrent une interruption. Pour comparer des signaux acquis selon des calendriers différents, harmoniser l’échantillonnage avant de calculer ces indices.

## Points d’entrée Python

- `agrisimu.spectral_analysis.spectral_analysis_matrix(S, dt)` prend une matrice de forme `(n_points, n_dates)` et un pas en secondes. Elle renvoie les fréquences en hertz et la puissance par point.
- `agrisimu.stable_phases.segment_stable_phases(S, timeline, dt)` prend la même forme de matrice, un `DatetimeIndex` et un pas nominal. Le calcul des durées repose sur les dates et renvoie des minutes. Les bornes de fin des segments sont exclusives.
- `agrisimu.irradiance_processing.compare_clusters(groupes, output=...)` prend un dictionnaire de groupes, chacun contenant ses points, écrit des figures et renvoie une table d’indices.
- `agrisimu.irradiance_processing.analyze_irradiance_data(points, output=...)` génère les cartes d’un ensemble de points. `imgsize` fixe la taille en cellules `(lignes, colonnes)` et `panels` permet de superposer les contours projetés des panneaux.
- `agrisimu.meteo_plots.plot_diurnal_heatmap(table, propname=...)` prend un tableau temporel et renvoie une figure et ses axes.

Pour les comparaisons exportées par le pipeline historique, les colonnes CSV `row` et `col` reprennent respectivement la première et la seconde coordonnée du dictionnaire. Cette convention est conservée pour la jointure avec les traitements et les mesures de rendement Camargue.
