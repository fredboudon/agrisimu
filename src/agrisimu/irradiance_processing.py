"""Comparaison de séries d’irradiance, indicateurs et graphiques."""

from math import floor
import pickle
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .data_util import dataframe_toarray, toarray, toimage, toimage_with_violin
from .spectral_analysis import (
    compute_bands_energy,
    detect_spectral_bands,
    plot_all_spectra,
    plot_mean_spectrum,
    plot_mean_spectrum_with_bands,
    plot_reference_band_energy,
    plot_spectral_map,
    plot_spectrum_in_period,
    spectral_analysis_matrix,
    spectral_intermittence_index,
)
from .stable_phases import (
    compute_phase_indices_per_plot,
    plot_stable_phase,
    plot_stable_phase_distribution_grouped,
    plot_stable_phase_histogram,
    segment_stable_phases,
)
from .statistics import (
    compute_pca_indices,
    compute_pca_with_clusters,
    plot_correlation_matrix,
    plot_pca_clusters,
    plot_pca_indices,
)
from .timeseries import (
    build_timeseries_matrix,
    cluster_distance_matrix,
    compute_distance_matrix,
    distance_between_points,
    distance_between_timeseries,
    distance_matrix,
    distance_matrix_fast,
    plot_distance_matrix,
)


def _plot_distributions(ax, groups, positions):
    """Trace les distributions disponibles aux positions des groupes."""
    valid = []
    for position, group in zip(positions, groups):
        values = np.asarray(group, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            valid.append((position, values))
    if valid:
        present, values = zip(*valid)
        ax.violinplot(values, positions=present, showmeans=True, showextrema=True)


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


def intermittence_char(S, timeline=None, dt=1, extra_params=None):
    print("run_stable_phase_detection")

    stable_phases = segment_stable_phases(S, timeline, dt, **(extra_params or {}))

    return {
        "durations": stable_phases[0],
        "means": stable_phases[1],
        "plots": stable_phases[2],
        "starts": stable_phases[3],
        "ends": stable_phases[4]
    }




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

def analyze_irradiance_data(plot_irradiances, meteo=None, output='analysis', imgsize=None, *, panels=None, examples=None, reference_bands=None):
    """Produit les cartes et les indices pour des séries synchronisées."""
    import os
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

    if len(timeline) < 2:
        raise ValueError('L’analyse nécessite au moins deux dates.')
    print('Timeline from', timeline[0], 'to', timeline[-1], 'with', len(timeline), 'points')
    dt = (timeline[1] - timeline[0]).total_seconds()
    print('Time step between measurements (seconds):', dt)


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
                if examples is not None:
                    coord, days = examples
                    if coord in coords:
                        available_days = set(timeline.strftime("%Y/%m/%d"))
                        for day in days:
                            if day in available_days:
                                plt.figure()
                                plot_stable_phase(data, stable_phases, timeline, day=day, plot=coords.index(coord))
                                plt.savefig(os.path.join(output, f'{prefix}stable_phase_example_{day.replace("/", "")}.png'))
                                plt.close()

                phase_indices = compute_phase_indices_per_plot(stable_phases['durations'], stable_phases['plots'], stable_phases['means'], timeline, n_plots=len(coords))
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
                spectrum_intermittence_index = True
                all_spectrum, spectral_map, spectrum_biggestfreq = [False]*3
                freqs, powers = spectral_analysis_matrix(data, dt=dt)
                log_scale_option = [False]
                if examples is not None and examples[0] in coords:
                    for log_scale in log_scale_option:
                        plot_spectrum_in_period(freqs, powers, coords.index(examples[0]), log_scale = log_scale)
                        plt.savefig(os.path.join(output, prefix+'power_spectrum_central'+('_log' if log_scale else '') +'.png'))
                        print('Saved power spectrum for', examples[0])
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
                if reference_bands and len(freqs) >= max(end for _, end in reference_bands['bands']):
                    ref_bands = reference_bands
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
                df_indices = pd.DataFrame({'Iq': Iq, 'If': If, 'It': It, 'Isp': Isp, 'Is': Is})
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
                    _plot_distributions(axes[i], [df_indices[col]], [1])
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

                df_indices["mean_S"] = mean_S
                if len(df_indices[columns].dropna()) < 2:
                    return df_indices
                pca_results = compute_pca_indices(df_indices, columns=columns)
                fig, ax = plot_pca_indices(pca_results, labels=latex_labels)
                plt.savefig(os.path.join(output,prefix+f'pca_intermittence_metrics.png'))
                plt.close()

                scores = pca_results["scores_df"]
                df_indices["I_global"] = scores["PC1"]
                df_indices["I_type"] = scores["PC2"]

                toimage_with_violin(toarray(coords, df_indices["I_global"], size=imgsize, boxes = panels), fname=os.path.join(output,prefix+f'global_intermittence_index1.png'), vmin=None)
                toimage_with_violin(toarray(coords, df_indices["I_type"], size=imgsize, boxes = panels), fname=os.path.join(output,prefix+f'global_intermittence_index2.png'), vmin=None)

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



def compare_clusters(irradiances_per_clusters, cluster_name = 'Cluster', cluster_title='per Cluster', cluster_order=None,output='clusters_comparison',imgsize=None, *, panels=None, reference_bands=None):
    import os
    from scipy.stats import kruskal, mannwhitneyu
    import itertools
    print(" *** Compare clusters ***")

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    if not irradiances_per_clusters:
        raise ValueError('Aucun groupe à comparer.')
    dt = {}

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
        if len(_timeline) < 2:
            raise ValueError(f'Le groupe {clusterid} contient moins de deux dates.')
        dt[clusterid] = (_timeline[1] - _timeline[0]).total_seconds()

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
        _plot_distributions(plt.gca(), mean_irradiance, positions)
        plt.xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        plt.title(f'Mean Irradiance {cluster_title}')
        plt.savefig(os.path.join(output, 'irradiance_mean.png'))
        plt.close()

        mean_shading = [Shadings[c].mean(axis=1) for c in clusters]
        plt.figure()
        _plot_distributions(plt.gca(), mean_shading, positions)
        plt.xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        plt.title(f'Mean Shading {cluster_title}')
        plt.savefig(os.path.join(output, 'shading_mean.png'))
        plt.close()  # Close the figure to avoid displaying it in interactive environments

        plt.figure()
        Iq = [estimate_var_metrics(DeltaShadings[c], dt[c])[1]*3600 for c in clusters]
        _plot_distributions(plt.gca(), Iq, positions)
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
                if reference_bands and len(freqs) >= max(end for _, end in reference_bands['bands']):
                    ref_bands = reference_bands
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
        _plot_distributions(plt.gca(), data, positions)
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
                phase_indices = compute_phase_indices_per_plot(stable_phases['durations'], stable_phases['plots'], stable_phases['means'], Timeline[clusterid], n_plots=len(Coords[clusterid]))
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
        _plot_distributions(plt.gca(), data, positions)
        plt.xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        plt.title(f'Stable Phase Lengths {cluster_title}')
        plt.savefig(os.path.join(output, 'stable_phase_length.png'))
        plt.close()

        data = [phase_fraction_per_cluster[c] for c in clusters]
        _plot_distributions(plt.gca(), data, positions)
        plt.xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        plt.title(f'Stable Phase Fractions {cluster_title}')
        plt.savefig(os.path.join(output, 'stable_fraction.png'))
        plt.close()

        data = [phase_mean_shading_per_cluster[c] for c in clusters]
        _plot_distributions(plt.gca(), data, positions)
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

        _plot_distributions(ax0, Iq, positions)
        ax0.set_xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        ax0.set_title(f'$I_{{q}}$ : Mean Absolute Shading Speed {cluster_title}', fontsize=10)

        data = [spectral_intermittence_index_per_cluster[c] for c in clusters]
        _plot_distributions(ax1, data, positions)
        ax1.set_xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        ax1.set_title(f'$I_{{sp}}$ : Spectral Intermittence Index {cluster_title}', fontsize=10)

        data = [phase_mean_length_per_cluster[c] for c in clusters]
        _plot_distributions(ax2, data, positions)
        ax2.set_xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        ax2.set_title(f'$I_{{t}}$ : Stable Phase Lengths {cluster_title}', fontsize=10)

        data = [phase_fraction_per_cluster[c] for c in clusters]
        _plot_distributions(ax3, data, positions)
        ax3.set_xticks(positions, [f'{cluster_name} {c}' for c in clusters])
        ax3.set_title(f'$I_{{f}}$ : Stable Phase Fractions {cluster_title}', fontsize=10)

        data = [phase_mean_shading_per_cluster[c] for c in clusters]
        _plot_distributions(ax4, data, positions)
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
        if len(df[index_columns].dropna()) < 2:
            return df

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
