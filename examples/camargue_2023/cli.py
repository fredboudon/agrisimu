"""Commandes de simulation et d’analyse pour Camargue 2023."""

import argparse
from pathlib import Path

from .paths import OUTPUT_DIR, REFERENCE_DIR

SCENARIOS = {"weather2023": None, "clear_sky": None, "cloudy": 0.3, "intermediate_sky": 0.5}
REFERENCE_BANDS = {
    "peaks": [401, 534, 134, 268, 668],
    "bands": [(395, 446), (526, 617), (129, 181), (262, 299), (661, 749)],
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
        raise ValueError("Aucun point sélectionné. Vérifiez la grille ou utilisez --full-grid.")

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
    parser = argparse.ArgumentParser(description="Étude agrivoltaïque de Camargue, campagne 2023")
    commands = parser.add_subparsers(dest="command", required=True)

    simulation = commands.add_parser("simulate", help="Simuler les cartes d’irradiance")
    simulation.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="weather2023")
    simulation.add_argument("--output", type=Path, help="Répertoire parent des scénarios")
    simulation.add_argument("--start", help="Date locale de début incluse (AAAA-MM-JJ)")
    simulation.add_argument("--end", help="Date locale de fin exclue (AAAA-MM-JJ)")
    simulation.add_argument("--jobs", type=int, default=1)
    simulation.set_defaults(run=simulate)

    analysis = commands.add_parser("analyze", help="Calculer et représenter les indices d’intermittence")
    analysis.add_argument("--mode", choices=("resolution", "treatments", "maps"), default="resolution")
    analysis.add_argument("--results", type=Path, default=OUTPUT_DIR / "result" / "weather2023")
    analysis.add_argument("--output", type=Path, help="Répertoire des figures et des indices")
    analysis.add_argument("--full-grid", action="store_true", help="Analyser tous les points simulés")
    analysis.set_defaults(run=analyze)

    yield_parser = commands.add_parser("yield", help="Relier les indices lumineux au rendement")
    yield_parser.add_argument("--indices", type=Path, default=REFERENCE_DIR / "comparison_results.csv")
    yield_parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "yield_analysis")
    yield_parser.set_defaults(run=run_yield)

    heatmap = commands.add_parser("heatmap", help="Tracer la carte jour-heure de l’irradiance")
    heatmap.add_argument("--scenario", choices=SCENARIOS, default="cloudy")
    heatmap.add_argument("--output", type=Path, default=OUTPUT_DIR / "daymap.png")
    heatmap.set_defaults(run=run_heatmap)

    args = parser.parse_args(argv)
    if args.command == "simulate":
        if args.jobs < 1:
            simulation.error("--jobs doit être positif.")
        if (args.start or args.end) and args.output is None:
            simulation.error("Indiquez --output pour distinguer une période partielle de la campagne complète.")
        args.output = args.output or OUTPUT_DIR / "result"
    if args.command == "analyze":
        args.output = args.output or OUTPUT_DIR / args.mode
    args.run(args)
