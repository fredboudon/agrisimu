"""Input and output paths for the Camargue case study."""

from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
DATA_DIR = CASE_DIR / "data"
REFERENCE_DIR = CASE_DIR / "reference"
OUTPUT_DIR = CASE_DIR / "outputs"

RESULT_DIR = OUTPUT_DIR / "result"
MEASURED_RESULTS = RESULT_DIR / "weather2023"
TREATMENT_OUTPUT = OUTPUT_DIR / "clusters_comparison" / "weather2023"


def yield_indices_path():
    """Prefer the treatment results and fall back to the bundled table."""
    treatment_indices = TREATMENT_OUTPUT / "comparison_results.csv"
    return treatment_indices if treatment_indices.is_file() else REFERENCE_DIR / "comparison_results.csv"
