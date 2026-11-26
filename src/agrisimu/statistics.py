"""Corrélations, analyses en composantes principales et figures associées."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
