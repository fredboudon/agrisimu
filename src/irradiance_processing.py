
from pandas import *
import numpy as np
from matplotlib import pyplot as plt
from math import *
import pickle
import importlib
#import data_util; importlib.reload(data_util)
from data_util import *
import time
import spectral_analysis; importlib.reload(spectral_analysis)
from spectral_analysis import *

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

import numpy as np
from pandas import date_range

import numpy as np
from pandas import date_range

def build_timeseries_matrix(plot_irradiances, property='irradiance',  defaultvalue=0):
    """
    Construit une matrice de timeseries à partir d'un dictionnaire de DataFrames.
    Supposition : toutes les séries ont les mêmes timestamps et mêmes valeurs manquantes.
    
    Retour :
        coords : liste des clés triées
        full_index : index temporel complet
        M : matrice (nc, nt) avec valeurs complétées
        mask_time : masque 1D (nt,) indiquant quelles colonnes étaient présentes dans l'index original
    """
    coords = sorted(plot_irradiances.keys())
    dfs = [plot_irradiances[c] for c in coords]

    full_index = dfs[0].index

    nt = len(full_index)
    nc = len(coords)

    # matrice finale
    M = np.zeros((nc, nt), dtype=float)

    for i, df in enumerate(dfs):
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
    coords, X = build_timeseries_matrix(plot_irradiances, property)
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



def estimate_meanvar_metrics(S):
    """
    Parameters
    ----------
    S : ndarray (n_squares, n_times)
        Taux d'ombrage (1 - irradiance_transmise)

    Returns
    -------
    mean_S : ndarray (n_squares,)
        Moyenne temporelle du taux d'ombrage

    var_S : ndarray (n_squares,)
        Variance temporelle du taux d'ombrage

    """

    # Moyenne d’ombrage
    mean_S = np.mean(S, axis=1)

    # Variabilité totale
    var_S = np.var(S, axis=1)


    return mean_S, var_S

def estimate_var_metrics(dS = None, dt = 1):
    # variance des variations temporelles (intermittence)
    var_dS = np.var(dS, axis=1)

    # moyenne absolue des variations temporelles (quantité d'intermittence)
    mean_abs_dS = np.mean(np.abs(dS), axis=1)/dt

    return var_dS, mean_abs_dS

def plot_speed_shading(S, dS, dt):
    plt.figure()

    plt.hexbin(S[:,1:].flatten(), (np.abs(np.diff(S, axis=1))/dt).flatten(), gridsize=50, cmap='jet',  norm='log', mincnt=1)
    plt.xlabel('Shading (S)')
    plt.ylabel('Absolute Speed (|dS|/dt)')
    plt.colorbar(label='Density')

import importlib
import stable_phases ; importlib.reload(stable_phases)
from stable_phases import *

def intermittence_char(S, timeline = None, dt = 1, extra_params={}):
    print("run_stable_phase_detection")

    stable_phases = segment_stable_phases(S, timeline, dt, **extra_params)

    return {
        "durations": stable_phases[0],
        "means": stable_phases[1],
        "plots": stable_phases[2],
        "starts": stable_phases[3],
        "ends": stable_phases[4]
    }

    return  stable_phases 



def cluster_metrics(n_cluster = 4, *args):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X = np.column_stack(args)
    X = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    labels = kmeans.fit_predict(X)
    return labels


def estimate_regression_residuals(S, meteo):
    from sklearn.linear_model import LinearRegression

    azimuth = np.radians(meteo['azimuth'].values)
    elevation = np.radians(meteo['elevation'].values)
    assert S.shape[1] == len(meteo), "Le nombre de points temporels dans S doit correspondre au nombre de lignes dans meteo"
    X = np.column_stack([
        np.sin(elevation),
        np.cos(elevation),
        np.sin(azimuth),
        np.cos(azimuth)
    ])

    model = LinearRegression()

    n_squares = S.shape[0]

    S_pred = np.zeros_like(S)
    residuals = np.zeros_like(S)
    R2 = np.zeros(n_squares)

    for i in range(n_squares):
        y = S[i, :]
        
        model.fit(X, y)
        
        S_pred[i, :] = model.predict(X)
        residuals[i, :] = y - S_pred[i, :]
        
        R2[i] = model.score(X, y)
    
    return S_pred, residuals, R2

def plot_correlation_matrix(df, columns, names = None, method="pearson", ax = None, figsize=(5, 4), cmap="coolwarm"):
    """
    Compute and plot correlation matrix.

    Parameters
    ----------
    df : pandas.DataFrame
    columns : list of str
        Columns to include (e.g. ["Iq", "Is", "It", "Isp"])
    method : str
        "pearson", "spearman", or "kendall"
    """
    if names is None:
        names = columns

    # --- sélection des colonnes ---
    data = df[columns]

    # --- corrélation ---
    corr = data.corr(method=method)

    # --- plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)

    # ticks
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    # rotation labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # annotations
    for i in range(len(columns)):
        for j in range(len(columns)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, label="Correlation")

    ax.set_title(f"{method.capitalize()} correlation")
    plt.tight_layout()

    return corr, fig


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_pca_indices(df, columns=("Iq", "Is", "It", "Isp"), dropna=True):
    """
    Compute PCA on selected index columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the indices.
    columns : list/tuple of str
        Columns used for PCA.
    dropna : bool
        If True, remove rows with missing values.

    Returns
    -------
    results : dict
        Contains:
        - scores_df: dataframe with PC coordinates
        - loadings_df: dataframe with variable loadings
        - explained_variance_ratio: explained variance ratio per PC
        - pca: fitted PCA object
        - scaler: fitted scaler
        - used_index: original dataframe index used in PCA
    """
    X = df.loc[:, columns].copy()

    if dropna:
        valid_mask = X.notna().all(axis=1)
        X = X.loc[valid_mask]
    else:
        if X.isna().any().any():
            raise ValueError("NaN values found. Use dropna=True or clean the data first.")

    used_index = X.index.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    pc_names = [f"PC{i+1}" for i in range(scores.shape[1])]

    scores_df = pd.DataFrame(scores, index=used_index, columns=pc_names)

    loadings = pca.components_.T
    loadings_df = pd.DataFrame(loadings, index=columns, columns=pc_names)

    return {
        "scores_df": scores_df,
        "loadings_df": loadings_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "pca": pca,
        "scaler": scaler,
        "used_index": used_index,
    }


def plot_pca_indices(pca_results, labels=None, color=None, figsize=(6, 5)):
    """
    Plot PCA scores on PC1-PC2 with variable loadings.

    Parameters
    ----------
    pca_results : dict
        Output of compute_pca_indices().
    labels : dict or None
        Optional mapping for variable labels, e.g.
        {"Iq": r"$I_q$", "Is": r"$I_s$", "It": r"$I_t$", "Isp": r"$I_{sp}$"}
    color : array-like or None
        Optional values to color points.
    figsize : tuple
        Figure size.
    """
    scores_df = pca_results["scores_df"]
    loadings_df = pca_results["loadings_df"]
    evr = pca_results["explained_variance_ratio"]

    if labels is None:
        labels = {c: c for c in loadings_df.index}

    fig, ax = plt.subplots(figsize=figsize)

    # --- scores
    if color is None:
        ax.scatter(scores_df["PC1"], scores_df["PC2"], alpha=0.7)
    else:
        c = np.asarray(color)
        if len(c) != len(scores_df):
            raise ValueError("color must have the same length as the PCA scores.")
        sc = ax.scatter(scores_df["PC1"], scores_df["PC2"], c=c, alpha=0.7)
        plt.colorbar(sc, ax=ax, fraction=0.046)

    # --- arrows for loadings
    arrow_scale = 2.0
    for var in loadings_df.index:
        x = loadings_df.loc[var, "PC1"] * arrow_scale
        y = loadings_df.loc[var, "PC2"] * arrow_scale

        ax.arrow(0, 0, x, y,
                 color="red", width=0.01, head_width=0.08,
                 length_includes_head=True)

        ax.text(x * 1.1, y * 1.1, labels.get(var, var),
                color="red", ha="center", va="center")

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)

    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}%)")
    ax.set_title("PCA of intermittency indices")

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax

def analyze_irradiance_data(plot_irradiances, meteo, output='analysis', imgsize = None):
    import os
    from importlib import reload
    from data_util import toarray, toimage
    from generateplot import retrieve_panel_projection
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    irradiance_metrics = False
    shading_metrics = True
    meanvar_metrics = True
    clustering = False
    stable_phases_metrics = True
    spectral_analysis = True
    meteo_dependency_analysis = False

    coords, timeline, Irradiance = build_timeseries_matrix(plot_irradiances, 'irradiance')
    coords, timeline, TrIrradiance = build_timeseries_matrix(plot_irradiances, 'TrIrradiance', defaultvalue=1)
    Shading = 1 - TrIrradiance 

    print('Timeline from', timeline[0], 'to', timeline[-1], 'with', len(timeline), 'points')
    dt = list(plot_irradiances.values())[0].index.diff().seconds[1]
    print('Time step between measurements (seconds):', dt)

    panels = retrieve_panel_projection()

    #import importlib
    #import smoothing ; importlib.reload(smoothing)
    #from smoothing import plot_smoothing_for_date
    #plot_smoothing_for_date(S=Shading, timeline=timeline, dt=dt, date='2023-06-21', plot_idx=coords.index((50,15)), gap_factor=1.5, uniform_sizes=(3, 5, 9), gaussian_sigmas=(1, 2), median_sizes=(3, 5), savgol_windows=(5, 9), savgol_polyorder=2, figsize=(12, 10), show_legend=True)
    #return
    
    toanalyse = []
    if irradiance_metrics:
        toanalyse.append(('irradiance_', Irradiance))
        
    if shading_metrics :
        toanalyse.append(('shading_', Shading))

    for prefix, data in toanalyse:
            prefix += 'total_'

            # subanalyses = [(gprefix+'total_', data)]
            # if meteo_dependency_analysis == True:
            #     print('Meteo dependency analysis')
            #     S_pred, S_residuals, R2 = estimate_regression_residuals(data, meteo)
            #     I_structure = 1 - R2
            #     toimage(toarray(coords, I_structure, size=imgsize, boxes = panels), fname=os.path.join(output,gprefix+f'structure_explained_variability.png'), vmin=0)
            #     subanalyses.append((gprefix+'structure_', S_residuals))
            #     subanalyses.append((gprefix+'sky_', S_pred))

            #     plt.figure()
            #     plt.hist(R2, bins=20)
            #     plt.xlabel("R² explained by sun position")
            #     plt.ylabel("Number of plots")
            #     plt.savefig(os.path.join(output, gprefix+'R2_distribution.png'))
            #     plt.close()  # Close the figure to avoid displaying it in interactive environments

            example_irr = data[:,0]
            toimage(toarray(coords, example_irr, size=imgsize, boxes = panels), fname=os.path.join(output,prefix+'example.png'), vmin = 0)

            #for prefix, data in subanalyses:
            print( '********', prefix)

            deltaData = np.diff(data, axis=1)
            
            if meanvar_metrics == True:
                print('Mean-Var metrics')
                mean_S, var_S = estimate_meanvar_metrics(data)
                var_dS, mean_abs_dS = estimate_var_metrics(deltaData, dt)
                toimage_with_violin(toarray(coords, mean_S, size=imgsize, boxes = panels), fname=os.path.join(output,prefix+'mean.png'), vmin=None)
                toimage_with_violin(toarray(coords, var_S, size=imgsize, boxes = panels), fname=os.path.join(output,prefix+'var.png'), vmin=None) #, vmin=0)
                toimage_with_violin(toarray(coords, var_dS, size=imgsize, boxes = panels), fname=os.path.join(output,prefix+'delta_var.png'), vmin=None)
                toimage_with_violin(toarray(coords, mean_abs_dS, size=imgsize, boxes = panels), fname=os.path.join(output,prefix+'delta_mean_abs.png'), vmin=None) #, vmin=0)
                Iq = mean_abs_dS * 3600 # express in "per hour"
                plt.figure()
                plot_speed_shading(data, deltaData, dt)
                plt.savefig(os.path.join(output, prefix+'delta_abs_over_data.png'))
                plt.close()  # Close the figure to avoid displaying it in interactive environments

            if stable_phases_metrics == True:
                print("Intermittence and stable phases analysis")
                timer = time.time()
                stable_phases_cache = os.path.join(output, prefix+'stable_phase_distribution.pkl')
                if os.path.exists(stable_phases_cache):
                     stable_phases = pickle.load(open(stable_phases_cache,'rb'))
                else:
                    stable_phases = intermittence_char(data,  timeline, dt)
                    pickle.dump(stable_phases, open(stable_phases_cache,'wb'))
                print('Found', len(stable_phases["durations"]), 'stable phases in', time.time() - timer, 'seconds')
                plt.figure()
                plot_stable_phase_histogram(stable_phases, dt)
                plt.savefig(os.path.join(output, prefix+'stable_phase_hist.png'))
                plt.close()  # Close the figure to avoid displaying it in interactive environments
                plt.figure()
                plot_stable_phase_distribution_grouped(stable_phases)
                plt.savefig(os.path.join(output, prefix+'stable_phase_distribution_grouped.png'))
                plt.close()  # Close the figure to avoid displaying it in interactive environments
                #plt.figure()
                #plot_stable_phase_distribution(stable_phases)
                #plt.savefig(os.path.join(output, prefix+'stable_phase_distribution.png'))
                #plt.close()  # Close the figure to avoid displaying it in interactive environments
                #plt.figure()                
                #plot_feature_space_with_kmeans(stable_phases['std_rel_all'], stable_phases['slope_all'], stable_phases['kmeans'],
                #                  std_thresh=stable_phases['std_thresh'], slope_thresh=stable_phases['slope_thresh'])
                #plt.savefig(os.path.join(output, prefix+'stable_phase_features.png'))
                #plt.close()  # Close the figure to avoid displaying it in interactive environments
                for day in ['2023/05/21', '2023/06/07', '2023/06/21', '2023/07/07', '2023/07/21', '2023/08/07', '2023/08/21', '2023/09/07', '2023/09/21', '2023/10/09']:
                     plt.figure()
                     plot_stable_phase(data, stable_phases, timeline, day=day, plot=coords.index((50,15)))
                     plt.savefig(os.path.join(output, prefix+f'stable_phase_example_{day.replace("/","")}.png'))
                     plt.close()  # Close the figure to avoid displaying it in interactive environments


                phase_indices = compute_phase_indices_per_plot(stable_phases['durations'], stable_phases['plots'], stable_phases['means'], timeline)
                toimage_with_violin(dataframe_toarray(phase_indices, coords, 'stable_fraction',size=imgsize, boxes = panels), fname=os.path.join(output,prefix+'stable_phase_fraction.png'))
                If = phase_indices['stable_fraction']

                toimage_with_violin(dataframe_toarray(phase_indices, coords, 'mean_phase_duration',size=imgsize, boxes = panels), fname=os.path.join(output,prefix+'stable_phase_mean_duration.png'))
                It = phase_indices['mean_phase_duration']


                toimage_with_violin(dataframe_toarray(phase_indices, coords, 'mean_phase_shading',size=imgsize, boxes = panels), fname=os.path.join(output,prefix+'stable_phase_mean_shading.png'))
                Is = phase_indices['mean_phase_shading']


                if clustering == True:
                    labels = cluster_metrics(4, mean_S, var_S, var_dS)
                    toimage(toarray(coords, labels, size=imgsize, boxes = panels), fname=os.path.join(output,prefix+'clustering.png'), vmin=0, vmax=3, cmap='tab10')
            
            if clustering == True:
                print('Distance matrix analysis')
                labels = cluster_distance_matrix(coords, distance_matrix_fast(data), n_clusters=4)
                toimage(toarray(coords, labels, size=imgsize, boxes = panels), fname=os.path.join(output,prefix+'distance_clustering.png'), vmin=0, vmax=3, cmap='tab10')

                print('Derivative distance matrix analysis')
                labels = cluster_distance_matrix(coords, distance_matrix_fast(deltaData), n_clusters=4)
                toimage(toarray(coords, labels, size=imgsize, boxes = panels), fname=os.path.join(output,prefix+'delta_distance_clustering.png'), vmin=0, vmax=3, cmap='tab10')

            if spectral_analysis == True:
                print('Spectral analysis')
                spectrum_example,  spectrum_intermittence_index, reference_bands = [True]*3
                all_spectrum, spectral_map, spectrum_biggestfreq = [False]*3
                freqs, powers = spectral_analysis_matrix(data, dt=dt)
                log_scale_option = [False]
                if spectrum_example:
                    for log_scale in log_scale_option:
                        plot_spectrum_in_period(freqs, powers, coords.index((50,15)), log_scale = log_scale)
                        plt.savefig(os.path.join(output, prefix+'power_spectrum_central'+('_log' if log_scale else '') +'.png'))
                        print('Saved power spectrum for central plot (50,15)')
                        plt.close()  # Close the figure to avoid displaying it in interactive environments
                for log_scale in log_scale_option:
                    plot_mean_spectrum(freqs, powers, log_scale = log_scale)
                    plt.savefig(os.path.join(output, prefix+'mean_spectrum'+('_log' if log_scale else '') +'.png'))
                    plt.close()
                print('Saved mean power spectrum')
                if all_spectrum:
                    for log_scale in log_scale_option:
                        plot_all_spectra(freqs, powers, log_scale = log_scale)
                        plt.savefig(os.path.join(output, prefix+'all_spectra'+('_log' if log_scale else '') +'.png'))
                        plt.close()
                        print('Saved all spectra')
                if spectral_map:
                    for log_scale in log_scale_option:
                        plot_spectral_map(freqs, powers, log_scale = log_scale)
                        plt.savefig(os.path.join(output, prefix+'spectral_map'+('_log' if log_scale else '') +'.png'))
                        plt.close()
                    print('Saved spectral map')
                if reference_bands:
                    ref_bands = {'peaks': array([401, 534, 134, 268, 668]), 'bands': [(395, 446), (526, 617), (129, 181), (262, 299), (661, 749)]}
                    energy = compute_bands_energy(powers, ref_bands['bands'])
                    plot_reference_band_energy(ref_bands, freqs, energy)
                    plt.savefig(os.path.join(output, prefix+'spectral_reference_bands.png'))
                    plt.close()
                    print('Saved spectral reference bands energy plot:'+repr(prefix+'spectral_reference_bands.png'))

                spectral_bands = detect_spectral_bands(freqs, powers)
                if spectral_bands:
                    plot_mean_spectrum_with_bands(freqs, powers, spectral_bands)
                    plt.savefig(os.path.join(output, prefix+'spectral_bands.png'))
                    plt.close()
                    print('Saved spectral bands plot')
                    print('Identified band frequencies:', spectral_bands['peaks'])
                    for i, (peak, bands) in enumerate(zip(spectral_bands['peaks'],spectral_bands['bands'])):
                        period_hours = 1 / freqs[peak] / 3600
                        period_minutes =  int(floor((period_hours % 1) * 60))
                        period_hours = int(floor(period_hours))
                        print(f'Frequency {i} index : {peak}, Band : {bands[0],bands[1]} Period (hours): {period_hours}h{str(period_minutes).zfill(2)}')
                        toimage(toarray(coords, powers[:, bands[0]:bands[1]].mean(axis=1), size=imgsize, boxes = panels), fname=os.path.join(output,prefix+f'power_peak_band_{i}_freq_{peak}_period_{period_hours}h{str(period_minutes).zfill(2)}.png'), vmin=0)
                if spectrum_biggestfreq:
                    biggestfreq = list(sorted(enumerate(powers.mean(axis=0)),key=lambda x : x[1], reverse=True))[:5]
                    print('Biggest frequencies:', biggestfreq)
                    for freq_index, power_value in biggestfreq:
                        period_hours = 1 / freqs[freq_index] / 3600
                        period_minutes =  int(floor((period_hours % 1) * 60))
                        period_hours = int(floor(period_hours))
                        print(f'Frequency index: {freq_index}, Period (hours): {period_hours}h{str(period_minutes).zfill(2)}, Power: {power_value:.4f}')
                        toimage(toarray(coords, powers[:, freq_index], size=imgsize, boxes = panels), fname=os.path.join(output,prefix+f'power_freq_{freq_index}_period_{period_hours}h{str(period_minutes).zfill(2)}.png'), vmin=0)
                if spectrum_intermittence_index:
                        sp_index = spectral_intermittence_index(freqs, powers)
                        toimage_with_violin(toarray(coords, sp_index, size=imgsize, boxes = panels), fname=os.path.join(output,prefix+f'spectral_intermittence_index.png'), vmin=None)
                        Isp = sp_index
            if  meanvar_metrics and stable_phases_metrics and  spectral_analysis :
                df_indices = DataFrame({'Iq': Iq, 'If': If, 'It': It, 'Isp': Isp, 'Is': Is})
                columns = ['Iq', 'Isp', 'If', 'It', 'Is']
                latex_labels = {
                    "Iq": r"$I_q$",
                    "If": r"$I_f$",
                    "It": r"$I_t$",
                    "Is": r"$I_s$",
                    "Isp": r"$I_{sp}$"
                }

                fig = plt.figure(figsize=(12, 8))

                # Ligne du haut : 2 figures centrées
                ax0 = plt.subplot2grid((2, 6), (0, 1), colspan=2)  # colonnes 1-2
                ax1 = plt.subplot2grid((2, 6), (0, 3), colspan=2)  # colonnes 3-4

                # Ligne du bas : 3 figures réparties
                ax2 = plt.subplot2grid((2, 6), (1, 0), colspan=2)  # colonnes 0-1
                ax3 = plt.subplot2grid((2, 6), (1, 2), colspan=2)  # colonnes 2-3
                ax4 = plt.subplot2grid((2, 6), (1, 4), colspan=2)  # colonnes 4-5

                axes = [ax0, ax1, ax2, ax3, ax4]

                for i, col in enumerate(columns):
                    axes[i].violinplot(
                        [df_indices[col].dropna()],
                        positions=[1],
                        showmeans=True,
                        showextrema=True
                    )
                    axes[i].set_title(latex_labels[col])
                    axes[i].set_xticks([1])
                    axes[i].set_xticklabels([])

                fig.suptitle("Distribution of intermittence indices", fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(output,prefix+f'intermittence_metrics_distributions.png'))
                plt.close()

                plot_correlation_matrix(df_indices, columns=columns, names=[latex_labels[col] for col in columns], method="spearman")
                plt.savefig(os.path.join(output,prefix+f'correlation_intermittence_metrics.png'))
                plt.close()

                pca_results = compute_pca_indices(df_indices, columns=columns)
                fig, ax = plot_pca_indices(pca_results, labels=latex_labels)
                plt.savefig(os.path.join(output,prefix+f'pca_intermittence_metrics.png'))
                plt.close()

                scores = pca_results["scores_df"]
                df_indices["I_global"] = scores["PC1"]
                df_indices["I_type"] = scores["PC2"]

                toimage_with_violin(toarray(coords, df_indices["I_global"], size=imgsize, boxes = panels), fname=os.path.join(output,prefix+f'global_intermittence_index1.png'), vmin=None)
                toimage_with_violin(toarray(coords, df_indices["I_type"], size=imgsize, boxes = panels), fname=os.path.join(output,prefix+f'global_intermittence_index2.png'), vmin=None)

                df_indices["mean_S"] = mean_S

                loadings = pca_results["loadings_df"]

                pc1_weights = loadings["PC1"]
                pc2_weights = loadings["PC2"]

                print("PCA Loadings for PC1 (global intermittence index):")
                print(pc1_weights)
                print("PCA Loadings for PC2 (type intermittence index):")
                print(pc2_weights)

                return df_indices


def plot_histogram_comparison(data_per_cluster):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scikit_posthocs as sp
    from scipy.stats import kruskal

    # --- Préparation données ---
    clusters = sorted(data_per_cluster.keys())
    data = [data_per_cluster[c].mean(axis=1) for c in clusters]

    # format long pour seaborn
    values = np.concatenate(data)
    groups = np.concatenate([[f'Cluster {c}'] * len(d) for c, d in zip(clusters, data)])

    # --- Style publication ---
    sns.set(style="whitegrid", context="paper", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(6, 4))

    # --- Violin plot ---
    sns.violinplot(x=groups, y=values,
                inner="quartile",
                linewidth=1,
                cut=0,
                ax=ax)

    # --- Test global ---
    H, p_global = kruskal(*data)

    # --- Post-hoc Dunn avec correction FDR ---
    p_matrix = sp.posthoc_dunn(data, p_adjust='fdr_bh')

    # --- Fonction annotation ---
    def significance_label(p):
        if p < 1e-3:
            return '***'
        elif p < 1e-2:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return None  # on n'affiche pas les ns

    # --- Ajout annotations ---
    y_max = values.max()
    y_min = values.min()
    height = y_max - y_min
    offset = height * 0.05
    current_y = y_max + offset

    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            p = p_matrix.iloc[i, j]
            label = significance_label(p)

            if label is None:
                continue

            x1, x2 = i, j

            ax.plot([x1, x1, x2, x2],
                    [current_y, current_y+offset, current_y+offset, current_y],
                    lw=1, c='black')

            ax.text((x1 + x2) / 2,
                    current_y + offset,
                    label,
                    ha='center', va='bottom')

            current_y += offset * 1.8  # espace vertical

    # --- Labels ---
    ax.set_title(f"Cluster comparison (Kruskal-Wallis p = {p_global:.2e})")

    sns.despine()
    plt.tight_layout()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_pca_with_clusters(df, index_columns, cluster_column="cluster", dropna=True):
    """
    Compute PCA on selected index columns and keep cluster labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the index columns and the cluster column.
    index_columns : list of str
        Names of the columns used for PCA.
    cluster_column : str
        Name of the column containing cluster labels.
    dropna : bool
        If True, remove rows with missing values in selected columns.

    Returns
    -------
    results : dict
        Contains:
        - scores_df: dataframe with PCA coordinates and cluster labels
        - loadings_df: dataframe with variable loadings
        - explained_variance_ratio: explained variance ratio
        - pca: fitted PCA object
        - scaler: fitted scaler
    """
    cols = list(index_columns) + [cluster_column]
    data = df.loc[:, cols].copy()

    if dropna:
        data = data.dropna(subset=cols)
    else:
        if data[cols].isna().any().any():
            raise ValueError("NaN values found. Use dropna=True or clean the dataframe first.")

    X = data[index_columns].to_numpy(dtype=float)
    clusters = data[cluster_column].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    pc_names = [f"PC{i+1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, columns=pc_names, index=data.index)
    scores_df[cluster_column] = clusters

    loadings = pca.components_.T
    loadings_df = pd.DataFrame(loadings, index=index_columns, columns=pc_names)

    return {
        "scores_df": scores_df,
        "loadings_df": loadings_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "pca": pca,
        "scaler": scaler,
    }


def plot_pca_clusters(
    pca_results,
    cluster_column="cluster",
    labels=None,
    figsize=(7, 6),
    arrow_scale=2.0,
    cluster_label = 'Cluster',
    cluster_title = 'Cluster',
    alpha=0.75
):
    """
    Plot PCA scores on PC1-PC2 colored by cluster, with variable loadings.
    """
    scores_df = pca_results["scores_df"]
    loadings_df = pca_results["loadings_df"]
    evr = pca_results["explained_variance_ratio"]

    if labels is None:
        labels = {c: c for c in loadings_df.index}

    fig, ax = plt.subplots(figsize=figsize)

    clusters = pd.unique(scores_df[cluster_column])
    clusters = sorted(clusters)

    # scatter by cluster
    for cl in clusters:
        sub = scores_df[scores_df[cluster_column] == cl]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            label=f"{cluster_label} {cl}",
            alpha=alpha
        )

    # loading arrows
    for var in loadings_df.index:
        x = loadings_df.loc[var, "PC1"] * arrow_scale
        y = loadings_df.loc[var, "PC2"] * arrow_scale

        ax.arrow(
            0, 0, x, y,
            color="red",
            width=0.01,
            head_width=0.08,
            length_includes_head=True
        )
        ax.text(
            x * 1.12, y * 1.12,
            labels.get(var, var),
            color="red",
            ha="center", va="center"
        )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)

    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}%)")
    ax.set_title(f"PCA of indices colored by {cluster_title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax

def compare_treatments(irradiances, clusters_file ='clusters.csv', output='clusters_comparison',imgsize = (30,100)):
    irradiances_per_clusters = sort_irradiances_per_cluster_from_file(irradiances, clusters_file, True)
    return compare_clusters(irradiances_per_clusters, cluster_name = 'T', cluster_title='per Treatment', output=output, imgsize=imgsize)

def compare_clusters(irradiances_per_clusters, cluster_name = 'Cluster', cluster_title='per Cluster', cluster_order=None,output='clusters_comparison',imgsize = (30,100)):
    import os
    from data_util import toarray, toimage
    from generateplot import retrieve_panel_projection
    from scipy.stats import kruskal, mannwhitneyu
    import itertools
    print(" *** Compare clusters ***")

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    panels = retrieve_panel_projection()
    dt = {key : list(irradiances_per_clusters[key].values())[0].index.diff().seconds[1] for key in irradiances_per_clusters.keys()}
    print('Time step between measurements (seconds):', dt)

    meanvar_metrics = True
    stable_phases_metrics = True
    spectral_analysis = True


    Irradiances = {}
    Shadings = {}
    DeltaShadings = {}
    Coords = {}
    Timeline = {}

    for clusterid, irradiances in irradiances_per_clusters.items():
        _coords, _timeline, Irradiance = build_timeseries_matrix(irradiances, 'irradiance')
        _coords, _timeline, TrIrradiance = build_timeseries_matrix(irradiances, 'TrIrradiance', defaultvalue=1)
        Shading = 1 - TrIrradiance 
        Irradiances[clusterid] = Irradiance
        Shadings[clusterid] = Shading
        DeltaShadings[clusterid] = np.diff(Shading, axis=1)
        Coords[clusterid] = _coords
        Timeline[clusterid] = _timeline

    if cluster_order is None:
        clusters = sorted(Irradiances.keys())
    else:
        clusters = [c for c in cluster_order if c in Irradiances]   
    positions = range(1, len(clusters) + 1)
    print('Comparing clusters:', clusters)
    print([(c, len(irradiances_per_clusters[c])) for c in clusters])
    
    if meanvar_metrics == True:
        mean_irradiance = [Irradiances[c].mean(axis=1) for c in clusters]


        plt.figure()

        # --- Violin plot ---
        plt.violinplot(mean_irradiance, positions=positions, showmeans=True, showextrema=True)
        plt.xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        plt.title(f'Mean Irradiance {cluster_title}')
        plt.savefig(os.path.join(output, 'irradiance_mean.png'))
        plt.close()

        mean_shading = [Shadings[c].mean(axis=1) for c in clusters]
        plt.figure()
        plt.violinplot(mean_shading, positions=positions, showmeans=True, showextrema=True)
        plt.xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        plt.title(f'Mean Shading {cluster_title}')
        plt.savefig(os.path.join(output, 'shading_mean.png'))
        plt.close()  # Close the figure to avoid displaying it in interactive environments

        plt.figure()
        Iq = [estimate_var_metrics(DeltaShadings[c], dt[c])[1]*3600 for c in clusters]
        plt.violinplot(Iq, positions=positions, showmeans=True, showextrema=True)
        plt.xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        plt.title(f'Mean Absolute Delta Shading {cluster_title}')
        plt.savefig(os.path.join(output, 'delta_shading_mean_abs.png'))
        plt.close()  # Close the figure to avoid displaying it in interactive environments

    if spectral_analysis == True:
        spectral_intermittence_index_per_cluster = {}
        spectral_bands_analysis = False
        for clusterid, shading in Shadings.items():
                print('Spectral analysis')
                freqs, powers = spectral_analysis_matrix(shading, dt=dt[clusterid])

                #plot_mean_spectrum(freqs, powers)
                #plt.savefig(os.path.join(output, 'mean_spectrum_cluster_{}.png').format(clusterid))
                #plt.close()
                #print('Saved mean power spectrum')
                ref_bands = {'peaks': array([401, 534, 134, 268, 668]), 'bands': [(395, 446), (526, 617), (129, 181), (262, 299), (661, 749)]}
                energy = compute_bands_energy(powers, ref_bands['bands'])
                plot_reference_band_energy(ref_bands, freqs, energy)
                plt.savefig(os.path.join(output, 'spectral_reference_bands_cluster_{}.png').format(clusterid))
                plt.close()
                print('Saved spectral reference bands energy plot:'+repr('spectral_reference_bands_cluster_{}.png').format(clusterid))

                spectral_bands = detect_spectral_bands(freqs, powers)
                if spectral_bands_analysis and spectral_bands:
                    plot_mean_spectrum_with_bands(freqs, powers, spectral_bands)
                    plt.savefig(os.path.join(output, 'spectral_bands_cluster_{}.png').format(clusterid))
                    plt.close()
                    print('Saved spectral bands plot')
                    print('Identified band frequencies:', spectral_bands['peaks'])
                    for i, (peak, bands) in enumerate(zip(spectral_bands['peaks'],spectral_bands['bands'])):
                        period_hours = 1 / freqs[peak] / 3600
                        period_minutes =  int(floor((period_hours % 1) * 60))
                        period_hours = int(floor(period_hours))
                        print(f'Frequency {i} index : {peak}, Band : {bands[0],bands[1]} Period (hours): {period_hours}h{str(period_minutes).zfill(2)}')
                        toimage(toarray(Coords[clusterid], powers[:, bands[0]:bands[1]].mean(axis=1), size=imgsize, boxes = panels), fname=os.path.join(output,f'power_peak_cluster_{clusterid}_band_{i}_freq_{peak}_period_{period_hours}h{str(period_minutes).zfill(2)}.png'), vmin=0)
                sp_index = spectral_intermittence_index(freqs, powers)
                spectral_intermittence_index_per_cluster[clusterid] = sp_index.flatten()
        plt.figure()

        # --- Violin plot ---
        data = [spectral_intermittence_index_per_cluster[c] for c in clusters]
        plt.violinplot(data, positions=positions, showmeans=True, showextrema=True)
        plt.xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        plt.title(f'Spectral Intermittence Index {cluster_title}')
        plt.savefig(os.path.join(output, 'intermittence_index.png'))
        plt.close()

    if stable_phases_metrics == True:
        phase_length_per_cluster = {}
        
        phase_fraction_per_cluster = {}
        phase_mean_length_per_cluster = {}
        phase_mean_shading_per_cluster = {}
        for clusterid, shading in Shadings.items():
                cache_fname = os.path.join(output, f'stable_phases_cluster_{clusterid}.pkl')
                if os.path.exists(cache_fname):
                    stable_phases = pickle.load(open(cache_fname,'rb'))
                else:
                    stable_phases = intermittence_char(shading,  Timeline[clusterid], dt[clusterid])
                    pickle.dump(stable_phases, open(cache_fname,'wb'))
                print('Found', len(stable_phases["durations"]), 'stable phases')
                phase_length_per_cluster[clusterid] = stable_phases["durations"]
                phase_indices = compute_phase_indices_per_plot(stable_phases['durations'], stable_phases['plots'], stable_phases['means'], Timeline[clusterid])
                phase_fraction_per_cluster[clusterid] = phase_indices["stable_fraction"]
                phase_mean_length_per_cluster[clusterid] = phase_indices["mean_phase_duration"]
                phase_mean_shading_per_cluster[clusterid] = phase_indices["mean_phase_shading"]
                plt.figure()
                plot_stable_phase_histogram(stable_phases, dt[clusterid])
                plt.savefig(os.path.join(output, 'stable_phase_hist_cluster_{}.png').format(clusterid))
                plt.close()  # Close the figure to avoid displaying it in interactive environments
                plt.figure()
                plot_stable_phase_distribution_grouped(stable_phases)
                plt.savefig(os.path.join(output, 'stable_phase_distribution_cluster_{}.png').format(clusterid))
                plt.close()  # Close the figure to avoid displaying it in interactive environments
                plt.figure()
        # --- Violin plot ---
        data = [phase_length_per_cluster[c] for c in clusters]
        plt.violinplot(data, positions=positions, showmeans=True, showextrema=True)
        plt.xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        plt.title(f'Stable Phase Lengths {cluster_title}')
        plt.savefig(os.path.join(output, 'stable_phase_length.png'))
        plt.close()

        data = [phase_fraction_per_cluster[c] for c in clusters]
        plt.violinplot(data, positions=positions, showmeans=True, showextrema=True)
        plt.xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        plt.title(f'Stable Phase Fractions {cluster_title}')
        plt.savefig(os.path.join(output, 'stable_fraction.png'))
        plt.close()

        data = [phase_mean_shading_per_cluster[c] for c in clusters]
        plt.violinplot(data, positions=positions, showmeans=True, showextrema=True)
        plt.xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        plt.title(f'Stable Phase Mean Shading {cluster_title}')
        plt.savefig(os.path.join(output, 'stable_mean_shading.png'))
        plt.close()



    if meanvar_metrics == True and spectral_analysis == True and stable_phases_metrics == True:

        fig = plt.figure(figsize=(12, 8))

        # Ligne du haut : 2 figures centrées
        ax0 = plt.subplot2grid((2, 6), (0, 1), colspan=2)  # colonnes 1-2
        ax1 = plt.subplot2grid((2, 6), (0, 3), colspan=2)  # colonnes 3-4

        # Ligne du bas : 3 figures réparties
        ax2 = plt.subplot2grid((2, 6), (1, 0), colspan=2)  # colonnes 0-1
        ax3 = plt.subplot2grid((2, 6), (1, 2), colspan=2)  # colonnes 2-3
        ax4 = plt.subplot2grid((2, 6), (1, 4), colspan=2)  # colonnes 4-5

        ax0.violinplot(Iq, positions=positions, showmeans=True, showextrema=True)
        ax0.set_xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        ax0.set_title(f'$I_{{q}}$ : Mean Absolute Shading Speed {cluster_title}', fontsize=10)

        data = [spectral_intermittence_index_per_cluster[c] for c in clusters]
        ax1.violinplot(data, positions=positions, showmeans=True, showextrema=True)
        ax1.set_xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        ax1.set_title(f'$I_{{sp}}$ : Spectral Intermittence Index {cluster_title}', fontsize=10)

        data = [phase_mean_length_per_cluster[c] for c in clusters]
        ax2.violinplot(data, positions=positions, showmeans=True, showextrema=True)
        ax2.set_xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        ax2.set_title(f'$I_{{t}}$ : Stable Phase Lengths {cluster_title}', fontsize=10)

        data = [phase_fraction_per_cluster[c] for c in clusters]
        ax3.violinplot(data, positions=positions, showmeans=True, showextrema=True)
        ax3.set_xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        ax3.set_title(f'$I_{{f}}$ : Stable Phase Fractions {cluster_title}', fontsize=10)

        data = [phase_mean_shading_per_cluster[c] for c in clusters]
        ax4.violinplot(data, positions=positions, showmeans=True, showextrema=True)
        ax4.set_xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        ax4.set_title(f'$I_{{s}}$ : Stable Phase Mean Shading {cluster_title}', fontsize=10)

        fig.suptitle("Distribution of intermittence indices", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output,f'intermittence_metrics_distributions.png'))
        plt.close()




        df = {
            'Iq': np.concatenate(Iq),
            'If': np.concatenate([phase_fraction_per_cluster[clusterid] for clusterid in clusters]),
            'It': np.concatenate([phase_mean_length_per_cluster[clusterid] for clusterid in clusters]),
            'Isp': np.concatenate([spectral_intermittence_index_per_cluster[clusterid] for clusterid in clusters]),
            "Is" : np.concatenate([phase_mean_shading_per_cluster[clusterid] for clusterid in clusters]),
            "mean_irradiance" : np.concatenate(mean_irradiance),
            "mean_shading" : np.concatenate(mean_shading),
            "cluster" : np.concatenate([[clusterid] * len(phase_fraction_per_cluster[clusterid]) for clusterid in clusters]),
            "row" : np.concatenate([[x[0] for x in Coords[clusterid]] for clusterid in clusters]),
            "col" : np.concatenate([[x[1] for x in Coords[clusterid]] for clusterid in clusters])
        }
        df = pd.DataFrame(df)

        index_columns = ["Iq", "If", "It", "Isp", "Is"]   # adapte à tes colonnes
        cluster_column = "cluster"

        latex_labels = {
            "Iq": r"$I_q$",
            "If": r"$I_f$",
            "It": r"$I_t$",
            "Isp": r"$I_{sp}$",
            "Is": r"$I_{s}$",
            "mean_irradiance": r"$\overline{I}$",
            "mean_shading": r"$\overline{S}$",
        }

        pca_results = compute_pca_with_clusters(
            df,
            index_columns=index_columns,
            cluster_column=cluster_column
        )

        cluster_title_modified = cluster_title.replace("per ", '') if cluster_title is not None else ''
        fig, ax = plot_pca_clusters(
            pca_results,
            cluster_column=cluster_column,
            labels=latex_labels,
            cluster_label=cluster_name,
            cluster_title=cluster_title_modified
        )
        plt.savefig(os.path.join(output, 'pca_intermittence_index.png'))
        plt.close()

        index_columns = ["Iq", "If", "It", "Isp", "Is", "mean_irradiance", "mean_shading"]   # adapte à tes colonnes
        pca_results = compute_pca_with_clusters(
            df,
            index_columns=index_columns,
            cluster_column=cluster_column
        )

        fig, ax = plot_pca_clusters(
            pca_results,
            cluster_column=cluster_column,
            labels=latex_labels,
            cluster_label=cluster_name
        )
        plt.savefig(os.path.join(output, 'pca_all_index.png'))
        plt.close()

        if len(np.unique([len(irradiances_per_clusters[c]) for c in clusters])) == 1:
            #print({cid:len(Iqi) for cid,Iqi in zip(clusters,Iq)})
            fig = plt.figure(figsize=(15, 8))

            # Ligne du haut : 2 figures centrées
            ax0 = plt.subplot2grid((2, 6), (0, 1), colspan=2)  # colonnes 1-2
            ax1 = plt.subplot2grid((2, 6), (0, 3), colspan=2)  # colonnes 3-4

            # Ligne du bas : 3 figures réparties
            ax2 = plt.subplot2grid((2, 6), (1, 0), colspan=2)  # colonnes 0-1
            ax3 = plt.subplot2grid((2, 6), (1, 2), colspan=2)  # colonnes 2-3
            ax4 = plt.subplot2grid((2, 6), (1, 4), colspan=2)  # colonnes 4-5
            for indicename, df_indice, ax in [('Iq', pd.DataFrame({cid:Iqi for cid,Iqi in zip(clusters,Iq)}), ax0), 
                                        ('If', pd.DataFrame({cid:phase_fraction_per_cluster[cid] for cid in clusters}), ax3), 
                                        ('It', pd.DataFrame({cid:phase_mean_length_per_cluster[cid] for cid in clusters}), ax2), 
                                        ('Isp', pd.DataFrame({cid:spectral_intermittence_index_per_cluster[cid] for cid in clusters}), ax1), 
                                        ('Is', pd.DataFrame({cid:phase_mean_shading_per_cluster[cid] for cid in clusters}), ax4)]:
                plot_correlation_matrix(df_indice, columns=clusters, method="spearman", ax = ax)
                ax.set_title(latex_labels[indicename])
            fig.suptitle("Spearman correlation of intermittence indices", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output,f'correlation_intermittence_metrics.png'))
            plt.close()

            for indicename, df_indice in [('mean_irradiance', pd.DataFrame({cid:mean_irradiance_i for cid,mean_irradiance_i in zip(clusters, mean_irradiance)})), 
                                        ('mean_shading', pd.DataFrame({cid:mean_shading_i for cid,mean_shading_i in zip(clusters, mean_shading)}))]:
                print(f'Correlation matrix for {indicename}')
                print(df_indice)
                plot_correlation_matrix(df_indice, columns=clusters, method="spearman")
                plt.savefig(os.path.join(output,f'correlation_{indicename}_metrics.png'))
                plt.close()



        return df

if __name__ == '__main__':
    import os
    from meteo import *

    irradiances_per_plot_5min = get_irradiances_per_plot('result/weather2023')
    #analyze_irradiance_data(irradiances_per_plot_5min, meteo, output='analysis/weather2023')

    irradiances_per_plot_5min = filter_dict(irradiances_per_plot_5min)
    #analyze_irradiance_data(irradiances_per_plot_5min, meteo, output='analysis_filtered/weather2023',imgsize = (30,100))
    #df = compare_treatments(irradiances_per_plot_5min, output='clusters_comparison/weather2023',imgsize = (30,100))
    #df.to_csv(os.path.join('clusters_comparison/weather2023', 'comparison_results.csv'))

    #for suffix, _meteo in [ ('clear_sky', generate_meteo()), ('cloudy', generate_meteo(attenuation=0.3)), ('intermediate_sky', generate_meteo(attenuation=0.5))]:
    #    plot_irradiances = get_irradiances_per_plot('result/'+suffix)
    #    #analyze_irradiance_data(plot_irradiances, _meteo, output='analysis'+suffix)
    #    #analyze_irradiance_data(filter_dict(plot_irradiances), _meteo, output='analysis_filtered'+suffix,imgsize = (30,100))
    
    #data = dict([(name,filter_dict(get_irradiances_per_plot('result/'+suffix))) for suffix, name in [ ('weather2023', '2023'), ('clear_sky', 'clear'), ('cloudy', 'cloudy'), ('intermediate_sky', 'intermediate')]])
    #compare_clusters(data, cluster_name='', cluster_title = 'per Weather', cluster_order = ['2023', 'clear', 'intermediate', 'cloudy'], output='meteo_comparison',imgsize = (30,100))

    #irradiances_per_plot_5min = filter_dict(irradiances_per_plot_5min)
    def resample(data, target_length):
        """Resample une série temporelle à une longueur cible en utilisant l'interpolation linéaire"""
        def transform(df0, target_length):
            agg = {"index" : "first", "row" : "first", "column" : "first", "irradiance" : "mean", "TrIrradiance" :"mean"}
            df = df0.resample(target_length).agg(agg).dropna(subset=['irradiance'])
            for col in df.columns:
                if col  in ['index', 'row', 'column']:
                    df[col] = df[col].astype(df0[col].dtype)
            return df
        if target_length != '5min':
            return { key: transform(df, target_length) for key, df in data.items() }
        return data

    timesteps = ['5min', '15min', '30min', '1h']
    #stable_phases_params = {'std_thresh': 0.1, 'slope_thresh': 0.01, 'min_duration': 3, 'max_duration': 1000}
    data = dict([(name,resample(irradiances_per_plot_5min, name)) for  name in timesteps] )
    compare_clusters(data, cluster_name='', cluster_title = 'per Time Resolution', cluster_order = timesteps, output='timeresolution_comparison',imgsize = (30,100))