"""Matrices temporelles, distances et classification des points de la grille."""

import time

import numpy as np
from matplotlib import pyplot as plt

from .data_util import retrieve_irradiance_from_plot


def distance_between_points(dataframe, row1, col1, row2, col2, property='irradiance'):
    df1 = retrieve_irradiance_from_plot(dataframe, row1, col1)
    df2 = retrieve_irradiance_from_plot(dataframe, row2, col2)
    return distance_between_timeseries(df1, df2, property=property)

def distance_between_timeseries(df1, df2, property='irradiance'):
    import numpy as np
    if len(df1) == 0 or len(df2) == 0:
        return np.nan
    return (df1[property] - df2[property]).abs().mean()


def distance_matrix(plot_irradiances, property='irradiance'):
    nbrows = max([i for i,j in plot_irradiances.keys()])+1
    nbcols = max([j for i,j in plot_irradiances.keys()])+1
    coords = [(i, j) for i in range(nbcols) for j in range(nbrows)]
    N = len(coords)
    matrix = np.zeros((N, N), dtype=float)
    t = time.time()
    for a in range(N):
        print('\r', a, '/', N, end='', flush=True)
        col, row = coords[a]
        for b in range(a+1, N):
            row1, col1 = coords[b]
            dist = distance_between_timeseries(plot_irradiances[(col, row)], plot_irradiances[(col1, row1)], property=property)
            matrix[row*nbcols+col, row1*nbcols+col1] = dist
            matrix[row1*nbcols+col1, row*nbcols+col] = dist
    print('\n')
    print('Total time to compute distance matrix:', time.time() - t)
    return matrix

def build_timeseries_matrix(plot_irradiances, property='irradiance',  defaultvalue=0):
    """
    Construit une matrice de timeseries à partir d'un dictionnaire de DataFrames.
    Supposition : toutes les séries ont les mêmes timestamps et mêmes valeurs manquantes.

    Renvoie les coordonnées triées, l’index temporel partagé et une matrice
    de forme (nombre de points, nombre de dates).
    """
    if not plot_irradiances:
        raise ValueError("Au moins une série est nécessaire.")
    coords = sorted(plot_irradiances)
    dfs = [plot_irradiances[c] for c in coords]

    full_index = dfs[0].index

    nt = len(full_index)
    nc = len(coords)

    # matrice finale
    M = np.zeros((nc, nt), dtype=float)

    for i, df in enumerate(dfs):
        if not df.index.equals(full_index):
            raise ValueError(f"Les dates du point {coords[i]} diffèrent de la série de référence.")
        M[i, :] = df[property].values

    return coords, full_index, M

def distance_matrix_fast(X):
    import time

    N = X.shape[0]
    D = np.zeros((N, N))

    t = time.time()
    for a in range(N):
        diff = np.abs(X[a] - X[a+1:])
        D[a, a+1:] = diff.mean(axis=1)
        D[a+1:, a] = D[a, a+1:]

    print('Total time to compute distance matrix fast:', time.time() - t)
    return D

def compute_distance_matrix(plot_irradiances, property='irradiance'):
    coords, _, X = build_timeseries_matrix(plot_irradiances, property)
    D = distance_matrix_fast(X)
    return coords, D

def cluster_distance_matrix(coords, distance_matrix, n_clusters):
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(distance_matrix)
    return labels


def plot_distance_matrix(coords, distance_matrix, fname):
    import matplotlib.pyplot as plt
    plt.matshow(distance_matrix, cmap='jet',vmin = 0, vmax=None)
    #plt.xticks(ticks=np.arange(len(coords)), labels=[f'({c[0]},{c[1]})' for c in coords], rotation=90)
    #plt.yticks(ticks=np.arange(len(coords)), labels=[f'({c[0]},{c[1]})' for c in coords])
    plt.colorbar()
    plt.show(block=False)
    plt.savefig(fname)
    plt.close()
