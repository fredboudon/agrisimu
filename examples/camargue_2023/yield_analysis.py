"""Crop yield and light indices grouped by treatment and replicate."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from agrisimu.statistics import compute_pca_with_clusters, plot_pca_clusters

from .paths import DATA_DIR, OUTPUT_DIR, yield_indices_path


def analyze_yield(indices_path=None, output=OUTPUT_DIR / "yield_analysis"):
    """Fit the models and save the PCA plot."""
    if indices_path is None:
        indices_path = yield_indices_path()
    output = Path(output)
    df = pd.read_csv(DATA_DIR / 'clusters.csv',
        dtype={'Rep': int,'group_clearsky': str})                 # cluster results
    df2 = pd.read_csv(indices_path)      # light indices
    df3 = pd.read_csv(DATA_DIR / '2023_Yield_comp 1.csv',
        dtype={'Rep': int,'Group': str})        # yield results

    df = df[df['Selected'] == "selected"]
    df = df[df['Zone'] == "Sampling"]

    df = df[['Row', 'Col', 'Rep', 'group_clearsky']]

    df.rename(columns={
        'group_clearsky': 'Group',
        'Row': 'col',
        'Col': 'row'
    }, inplace=True)

    merged = pd.merge(
        df,
        df2,
        on=['row', 'col'],
        how='inner'
    )

    index_cols = ['Iq', 'If', 'It', 'Isp', 'Is', 'mean_irradiance','mean_shading']

    indices_mean = (
        merged
        .groupby(['Group', 'Rep'])[index_cols]
        .mean()
        .reset_index()
    )

    final_df = pd.merge(
        indices_mean,
        df3,
        on=['Group', 'Rep'],
        how='inner'
    )

    print(final_df)

    variables = [
        'Iq', 'If', 'It', 'Isp',
        'Is', 'mean_irradiance',
        'mean_shading'
    ]

    assert all(var in final_df.columns for var in variables), "Some variables are missing in the final dataframe."

    pca_results = compute_pca_with_clusters(
        final_df,
        index_columns=variables,
        cluster_column='Group'
    )

    latex_labels = {
        "Iq": r"$I_q$",
        "If": r"$I_f$",
        "It": r"$I_t$",
        "Isp": r"$I_{sp}$",
        "Is": r"$I_{s}$",
        "mean_irradiance": r"$\overline{I}$",
        "mean_shading": r"$\overline{S}$",
    }

    fig, ax = plot_pca_clusters(
        pca_results,
        cluster_column='Group',
        labels=latex_labels
    )
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'pca_intermittence_index.png')
    plt.close()

    model = smf.ols(
        "Yield_tha ~ mean_irradiance + mean_shading + Iq + If + It + Isp + Is",
        data=final_df
    )

    result = model.fit()

    print(result.summary())

    results = []

    for var in variables:

        model = smf.ols(
            f"Yield_tha ~ {var}",
            data=final_df
        )

        result = model.fit(reml=False)

        results.append({
            'Variable': var,
            'Coef': result.params[var],
            'pvalue': result.pvalues[var],
            'AIC': result.aic
        })

    results_df = pd.DataFrame(results)

    print(results_df.sort_values('pvalue'))

    print(final_df[variables].corr())

    X = final_df[variables]
    y = final_df['Yield_tha']

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(cv=4)

    lasso.fit(X_scaled, y)

    selected = pd.DataFrame(
        lasso.coef_,
        index=variables,
        columns=['LassoCoefficient']
    )

    selected.sort_values('LassoCoefficient', key=abs, ascending=False, inplace=True)

    print(selected)

    model = smf.ols(
        "Yield_tha ~ mean_irradiance + Iq",
        data=final_df
    )

    result = model.fit()

    print(result.summary())

    model = smf.ols(
        "Yield_tha ~ mean_irradiance",
        data=final_df
    )

    result = model.fit()

    print(result.summary())

    return final_df


if __name__ == "__main__":
    analyze_yield()
