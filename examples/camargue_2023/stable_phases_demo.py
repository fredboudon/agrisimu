"""Détection de phases sur un signal d’illustration."""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from agrisimu.stable_phases import segment_stable_phases
    from .paths import OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    nt = 500
    timeline = pd.date_range("2023-06-21 07:00", periods=nt, freq="2min")

    # Signal synthétique : deux phases stables + bruit + transitoire
    S_row = np.concatenate([
        np.full(80, 0.28) + rng.normal(0, 0.005, 80),   # phase stable ~0.28
        np.linspace(0.28, 0.15, 40) + rng.normal(0, 0.01, 40),  # transition
        rng.uniform(0.10, 0.42, 200),                   # phase variable
        np.full(80, 0.22) + rng.normal(0, 0.005, 80),   # phase stable ~0.22
        rng.uniform(0.05, 0.30, 100),                   # fin variable
    ])
    S = S_row[np.newaxis, :]  # shape (1, nt)

    durations, meanS, plots, starts, ends = segment_stable_phases(
        S, timeline, dt=2
    )

    t_min = (timeline - timeline[0]) / np.timedelta64(1, "m")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_min, S_row, color="steelblue", lw=0.8, label="Ombrage")
    for s, e, m in zip(starts, ends, meanS):
        ax.axhspan(m - 0.005, m + 0.005, xmin=t_min[s] / t_min[-1],
                   xmax=t_min[e - 1] / t_min[-1], alpha=0.15, color="red")
        ax.hlines(m, t_min[s], t_min[e - 1], colors="red", lw=2)
    ax.set_xlabel("Temps (min)")
    ax.set_ylabel("Ombrage")
    ax.set_title("Phases stables détectées")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_stable_phases.png", dpi=120)
    print(f"{len(starts)} phases stables détectées.")
    for k in range(len(starts)):
        print(f"  Segment {k+1}: t={t_min[starts[k]]:.0f}–{t_min[ends[k]-1]:.0f} min, "
              f"ombrage moyen={meanS[k]:.3f}, durée={durations[k]:.0f} min")
