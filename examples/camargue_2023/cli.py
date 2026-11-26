"""Simulation and analysis commands for Camargue 2023."""

import argparse
from pathlib import Path

from .paths import MEASURED_RESULTS, OUTPUT_DIR, RESULT_DIR, TREATMENT_OUTPUT, yield_indices_path

SCENARIOS = {"weather2023": None, "clear_sky": None, "cloudy": 0.3, "intermediate_sky": 0.5}
REFERENCE_BANDS = {
    "peaks": [401, 534, 134, 268, 668],
    "bands": [(395, 446), (526, 617), (129, 181), (262, 299), (661, 749)],
}
CAMPAIGN_ANALYSIS_OUTPUTS = {
    "resolution": OUTPUT_DIR / "timeresolution_comparison",
    "treatments": TREATMENT_OUTPUT,
    "maps": OUTPUT_DIR / "analysis_filtered" / "weather2023",
}
TIMESTEPS = ("5min", "15min", "30min", "1h")
SAMPLE_DAYS = (
    "2023/05/21", "2023/06/07", "2023/06/21", "2023/07/07", "2023/07/21",
    "2023/08/07", "2023/08/21", "2023/09/07", "2023/09/21", "2023/10/09",
)


def scenario_weather(name, measured):
    from .meteo import generate_meteo

    if name == "weather2023":
        return measured
    return generate_meteo(measured.index, attenuation=SCENARIOS[name])


def simulate(args):
    import pandas as pd

    from .lightsimulation import mt_process_light
    from .meteo import prepare_meteo

    measured = prepare_meteo()
    start = pd.Timestamp(args.start, tz=measured.index.tz) if args.start else None
    end = pd.Timestamp(args.end, tz=measured.index.tz) if args.end else None
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    for name in scenarios:
        weather = scenario_weather(name, measured)
        mt_process_light(weather, mindate=start, maxdate=end, outdir=args.output / name, nbjobs=args.jobs)


def resample_irradiances(plots, timestep):
    if timestep == "5min":
        return plots
    aggregation = {
        "index": "first", "row": "first", "column": "first",
        "irradiance": "mean", "TrIrradiance": "mean",
    }
    result = {}
    for coord, frame in plots.items():
        available = {col: method for col, method in aggregation.items() if col in frame.columns}
        sampled = frame.resample(timestep).agg(available).dropna(subset=["irradiance"])
        result[coord] = sampled.astype({col: frame[col].dtype for col in ("index", "row", "column") if col in sampled})
    return result


def analyze(args):
    from agrisimu.data_util import get_irradiances_per_plot
    from agrisimu.irradiance_processing import analyze_irradiance_data, compare_clusters

    from .data_util import filter_dict, sort_irradiances_per_cluster_from_file

    plots = get_irradiances_per_plot(args.results)
    if not args.full_grid:
        plots = filter_dict(plots)
    if not plots:
        raise ValueError("No grid points selected. Check the grid or use --full-grid.")

    if args.mode == "maps":
        from .generateplot import retrieve_panel_projection

        metrics = analyze_irradiance_data(
            plots, output=args.output, imgsize=(30, 100), panels=retrieve_panel_projection(),
            examples=((50, 15), SAMPLE_DAYS), reference_bands=REFERENCE_BANDS,
        )
        metrics["column"], metrics["row"] = zip(*sorted(plots))
    elif args.mode == "treatments":
        groups = sort_irradiances_per_cluster_from_file(plots)
        metrics = compare_clusters(groups, cluster_name="T", cluster_title="per Treatment", output=args.output,
                                   reference_bands=REFERENCE_BANDS)
    else:
        groups = {step: resample_irradiances(plots, step) for step in TIMESTEPS}
        metrics = compare_clusters(
            groups, cluster_name="", cluster_title="per Time Resolution",
            cluster_order=TIMESTEPS, output=args.output, reference_bands=REFERENCE_BANDS,
        )
    metrics.to_csv(args.output / "comparison_results.csv")


def run_yield(args):
    from .yield_analysis import analyze_yield

    analyze_yield(args.indices, args.output)


def run_heatmap(args):
    import matplotlib.pyplot as plt
    from agrisimu.meteo_plots import plot_diurnal_heatmap

    from .meteo import prepare_meteo

    weather = scenario_weather(args.scenario, prepare_meteo())
    fig, _ = plot_diurnal_heatmap(weather, propname="ghi")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Camargue 2023 agrivoltaic field study")
    commands = parser.add_subparsers(dest="command", required=True)

    simulation = commands.add_parser("simulate", help="Simulate irradiance maps")
    simulation.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="weather2023")
    simulation.add_argument("--output", type=Path, help="Parent directory for scenario outputs")
    simulation.add_argument("--start", help="Local start date, inclusive (YYYY-MM-DD)")
    simulation.add_argument("--end", help="Local end date, exclusive (YYYY-MM-DD)")
    simulation.add_argument("--jobs", type=int, default=1)
    simulation.set_defaults(run=simulate)

    analysis = commands.add_parser("analyze", help="Calculate and plot intermittency indices")
    analysis.add_argument("--mode", choices=("resolution", "treatments", "maps"), default="resolution")
    analysis.add_argument("--results", type=Path, default=MEASURED_RESULTS)
    analysis.add_argument("--output", type=Path, help="Output directory for figures and indices")
    analysis.add_argument("--full-grid", action="store_true", help="Include all simulated grid points")
    analysis.set_defaults(run=analyze)

    yield_parser = commands.add_parser("yield", help="Relate irradiance indices to crop yield")
    yield_parser.add_argument("--indices", type=Path, help="Path to a table of treatment indices")
    yield_parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "yield_analysis")
    yield_parser.set_defaults(run=run_yield)

    heatmap = commands.add_parser("heatmap", help="Plot irradiance by date and time of day")
    heatmap.add_argument("--scenario", choices=SCENARIOS, default="cloudy")
    heatmap.add_argument("--output", type=Path, default=OUTPUT_DIR / "daymap.png")
    heatmap.set_defaults(run=run_heatmap)

    args = parser.parse_args(argv)
    if args.command == "simulate":
        if args.jobs < 1:
            simulation.error("--jobs must be a positive integer.")
        if (args.start or args.end) and args.output is None:
            simulation.error("Use --output to save a partial simulation separately from the full campaign.")
        args.output = args.output or RESULT_DIR
    if args.command == "analyze" and args.output is None:
        if args.results.resolve() == MEASURED_RESULTS and not args.full_grid:
            args.output = CAMPAIGN_ANALYSIS_OUTPUTS[args.mode]
        else:
            args.output = OUTPUT_DIR / args.mode
    if args.command == "yield" and args.indices is None:
        args.indices = yield_indices_path()
    args.run(args)
