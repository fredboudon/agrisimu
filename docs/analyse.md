# Analyze irradiance time series

## Inputs

The reporting functions in `agrisimu.irradiance_processing` take pandas DataFrames indexed by timestamp and stored in a dictionary of the form `{(column, row): DataFrame}`. Grid coordinates are zero-based integers. All points in a group must share the same timestamps in chronological order. Each DataFrame requires two columns.

| Column | Definition |
| --- | --- |
| `irradiance` | Irradiance received at the simulated point, in W·m⁻² |
| `TrIrradiance` | Irradiance received at the simulated point divided by incident global horizontal irradiance (GHI) at the same timestamp |

Shading is calculated as `S = 1 - TrIrradiance`. For data from an external table, convert the timestamps to a `DatetimeIndex` and group the data by `column` and `row`. The `retrieve_irradiance_per_plot` function does this for a table with timestamps in the `date_time` column.

`get_irradiances_per_plot(path)` reads a simulation directory. It accepts tab-separated files named `simulation_<height>_<YYYY-MM-DD-HH-MM>.csv` and existing aggregates named `global_irradiances.pkl` or `global_irradiances.pkl.zip`. Height is expressed in meters, with an underscore as the decimal separator. Timestamps in TSV filenames are in local time and carry no time-zone information. Only load pickle files from trusted sources.

## Indices

| Column | Calculation |
| --- | --- |
| `mean_irradiance` | Mean simulated irradiance in W·m⁻² |
| `mean_shading` | Mean shading over time |
| `Iq` | Mean absolute change in shading between consecutive samples, scaled to an hourly rate |
| `Isp` | Fraction of spectral energy above the frequency corresponding to a six-hour period. A constant series has zero energy, so the index is undefined. |
| `If` | Total duration of stable phases divided by the time elapsed from the first to the last sample |
| `It` | Mean duration of stable phases in minutes |
| `Is` | Mean of the per-phase average shading levels across stable phases |

Stable-phase detection uses the rolling standard deviation of a Gaussian-smoothed signal and processes each day separately. The duration of a phase runs from its first to its last sample. `It` and `Is` are missing when no stable phase is detected. The reports run principal component analysis (PCA) using points with values for every index included in the PCA.

The reports infer the nominal sampling interval from the first two timestamps. The fast Fourier transform (FFT) treats successive samples as equally spaced, including when field data omit nighttime samples. The differences used by `Iq` also include sample pairs on either side of a gap. To compare series collected on different schedules, align their sampling before calculating these indices.

## Python entry points

- `agrisimu.spectral_analysis.spectral_analysis_matrix(S, dt)` takes a matrix of shape `(n_points, n_timestamps)` and a sampling interval in seconds. It returns frequencies in hertz and spectral power for each point.
- `agrisimu.stable_phases.segment_stable_phases(S, timeline, dt)` takes a matrix of the same shape, a `DatetimeIndex`, and a nominal sampling interval. It calculates durations in minutes from the timestamps. Segment end indices are exclusive.
- `agrisimu.irradiance_processing.compare_clusters(groups, output=...)` takes a dictionary of groups, each containing its grid points. It saves figures and returns a table of indices.
- `agrisimu.irradiance_processing.analyze_irradiance_data(points, output=...)` generates maps for a set of points. `imgsize` sets the grid dimensions in cells `(rows, columns)`, and `panels` overlays the projected panel outlines.
- `agrisimu.meteo_plots.plot_diurnal_heatmap(table, propname=...)` takes a time-indexed DataFrame and returns a figure and its axes.

In comparison CSVs from the original workflow, `row` and `col` store the first and second coordinates of the dictionary key, respectively. This convention is retained so the exported indices can be joined with the Camargue treatment and yield data.
