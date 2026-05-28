import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================

df = pd.read_csv('clusters.csv',
    dtype={'Rep': int,'group_clearsky': str})                 # cluster results
df2 = pd.read_csv('comparison_results.csv')      # light indices
df3 = pd.read_csv('2023_Yield_comp 1.csv',
    dtype={'Rep': int,'Group': str})        # yield results

#print(df2.head())

# =========================
# PREPARE DF1
# =========================

df = df[df['Selected'] == "selected"]
df = df[df['Zone'] == "Sampling"]

df = df[['Row', 'Col', 'Rep', 'group_clearsky']]

df.rename(columns={
    'group_clearsky': 'Group',
    'Row': 'col',
    'Col': 'row'
}, inplace=True)

#print(np.unique(df['row']))
#print(np.unique(df2['row']))
#print(np.unique(df['col']))
#print(np.unique(df2['col']))


#print(df.head())
#print(df2.head())
# =========================
# MERGE WITH LIGHT INDICES
# =========================
#print(df.columns)
#print(df2.columns)

# Merge row/col -> Group/Rep with indices
merged = pd.merge(
    df,
    df2,
    on=['row', 'col'],
    how='inner'
)

#print(merged.head())

# =========================
# COMPUTE MEAN INDICES
# =========================

# Columns containing indices
index_cols = ['Iq', 'If', 'It', 'Isp', 'Is', 'mean_irradiance','mean_shading']

# Mean indices for each Group/Rep
indices_mean = (
    merged
    .groupby(['Group', 'Rep'])[index_cols]
    .mean()
    .reset_index()
)

# =========================
# MERGE WITH YIELD DATA
# =========================
#print(df3.head())

final_df = pd.merge(
    indices_mean,
    df3,
    on=['Group', 'Rep'],
    how='inner'
)

# =========================
# RESULT
# =========================

print(final_df)

from irradiance_processing import compute_pca_with_clusters, plot_pca_clusters
import matplotlib.pyplot as plt
import os

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
if not os.path.exists('yield_analysis'):
    os.makedirs('yield_analysis')
plt.savefig(os.path.join('yield_analysis', 'pca_intermittence_index.png'))
plt.close()


import statsmodels.formula.api as smf

model = smf.ols(
    "Yield_tha ~ mean_irradiance + mean_shading + Iq + If + It + Isp + Is",
    data=final_df
)

result = model.fit()

print(result.summary())



import statsmodels.formula.api as smf



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


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

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