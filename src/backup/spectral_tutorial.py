import numpy as np
import matplotlib.pyplot as plt


def create_synthetic_irradiance():
    # 24 h avec un point toutes les 5 minutes
    t_hours = np.arange(0, 24, 5 / 60)

    # cycle journalier de base
    result = np.array([
        0.8 * np.sin((t - 6) * np.pi / 12) if 6 <= t <= 18 else 0
        for t in t_hours
    ])

    # deux épisodes d'ombrage
    result[int(9.5 * 12): int(10.5 * 12) + 1] = 0.1
    result[int(14 * 12): int(16 * 12) + 1] = 0.1

    return t_hours, result


def fft_analysis(signal, dt_seconds):
    freqs = np.fft.rfftfreq(len(signal), d=dt_seconds)
    fft_coeffs = np.fft.rfft(signal)
    return freqs, fft_coeffs


def reconstruct_component(k, fft_coeffs, n):
    """
    Reconstruit la contribution temporelle d'une composante FFT.
    """
    idx = np.arange(n)

    if k == 0:
        return np.full(n, np.real(fft_coeffs[0]) / n)

    # cas général pour rfft
    return 2 * np.real(fft_coeffs[k] * np.exp(2j * np.pi * k * idx / n)) / n


def period_from_freq(freq):
    if freq == 0:
        return np.inf
    return 1 / freq / 3600


def describe_component(period_h):
    if np.isinf(period_h):
        return "Niveau moyen du signal"
    elif period_h > 18:
        return "Forme journalière globale"
    elif period_h > 8:
        return "Variation lente intra-journée"
    elif period_h > 2:
        return "Modulation intermédiaire"
    else:
        return "Détails rapides / transitions"


def select_main_components(freqs, fft_coeffs, n_oscillatory=4):
    """
    Sélectionne :
    - la moyenne (k=0)
    - les composantes oscillantes les plus énergétiques
    """
    magnitudes = np.abs(fft_coeffs)

    # on ignore la moyenne pour classer les oscillantes
    sorted_idx = np.argsort(magnitudes[1:])[::-1] + 1
    main_idx = list(sorted_idx[:n_oscillatory])

    return [0] + main_idx


def plot_fft_tutorial(t_hours, signal, freqs, fft_coeffs, selected_indices):
    n = len(signal)
    cumulative = np.zeros(n)

    nrows = len(selected_indices) + 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=3,
        figsize=(15, 2.6 * nrows),
        sharex='col'
    )

    # --------------------------
    # Ligne 0 : signal initial
    # --------------------------
    axes[0, 0].plot(t_hours, signal, linewidth=2)
    axes[0, 0].set_title("Signal original")
    axes[0, 0].set_ylabel("Irradiance")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.0, 0.6,
        "Point de départ :\nle signal observé\nà décomposer.",
        fontsize=11,
        va="center"
    )

    # --------------------------
    # Lignes suivantes
    # --------------------------
    for i, k in enumerate(selected_indices, start=1):
        comp = reconstruct_component(k, fft_coeffs, n)
        cumulative += comp

        # infos
        if k == 0:
            label = "Moyenne"
            comment = "Décale tout le signal vers le haut ou le bas."
        else:
            period_h = period_from_freq(freqs[k])
            label = f"Période = {period_h:.2f} h"
            comment = describe_component(period_h)

        # Colonne 1 : composante seule
        axes[i, 0].plot(t_hours, comp, linewidth=2)
        axes[i, 0].set_ylabel(label)
        axes[i, 0].grid(True, alpha=0.3)

        # Colonne 2 : somme cumulée
        axes[i, 1].plot(t_hours, signal, color="lightgray", linewidth=2, label="Original")
        axes[i, 1].plot(t_hours, cumulative, linewidth=2, label="Somme cumulée")
        axes[i, 1].grid(True, alpha=0.3)

        # Colonne 3 : commentaire
        axes[i, 2].axis("off")
        axes[i, 2].text(
            0.0, 0.5,
            comment,
            fontsize=11,
            va="center"
        )

    axes[0, 1].set_title("Assemblage progressif")
    axes[0, 2].set_title("Interprétation")

    for ax in axes[-1, :2]:
        ax.set_xlabel("Temps (h)")

    # légende une seule fois
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_spectrum(freqs, fft_coeffs):
    magnitudes = np.abs(fft_coeffs)
    nonzero = freqs > 0
    periods = 1 / freqs[nonzero] / 3600

    plt.figure(figsize=(10, 4))
    plt.plot(periods, magnitudes[nonzero], "o-")
    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.title("Spectre de Fourier")
    plt.xlabel("Période (h)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_loglog_spectrum(freqs, fft_coeffs, 
                         use_period=True,
                         normalize=False,
                         annotate_peaks=True):
    """
    Affichage du spectre FFT en log-log.

    Parameters
    ----------
    freqs : ndarray
        Fréquences (Hz)
    fft_coeffs : ndarray
        Coefficients FFT (complexes)
    use_period : bool
        Si True : axe X en périodes (heures)
        Sinon : axe X en fréquences (Hz)
    normalize : bool
        Si True : normalise les amplitudes
    annotate_peaks : bool
        Ajoute des annotations simples (24h, 12h, etc.)
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # --- amplitude ---
    magnitudes = np.abs(fft_coeffs)

    # --- enlever fréquence nulle ---
    mask = freqs > 0
    freqs = freqs[mask]
    magnitudes = magnitudes[mask]

    # --- normalisation ---
    if normalize:
        magnitudes = magnitudes / np.max(magnitudes)

    # --- conversion en période ---
    if use_period:
        x = 1 / freqs / 3600  # en heures
        xlabel = "Période (heures)"
    else:
        x = freqs
        xlabel = "Fréquence (Hz)"

    # --- plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(x, magnitudes, linewidth=2)

    plt.xscale('log')
    plt.yscale('log')

    if use_period:
        plt.gca().invert_xaxis()  # grandes périodes à gauche

    plt.xlabel(xlabel)
    plt.ylabel("Amplitude")
    plt.title("Spectre FFT (log-log)")
    plt.grid(True, which="both", alpha=0.3)

    # --- ticks utiles ---
    if use_period:
        ticks = [1, 2, 4, 8, 12, 24]
        plt.xticks(ticks, labels=[f"{t}h" for t in ticks])

    # --- annotations simples ---
    if annotate_peaks and use_period:
        for p in [24, 12, 6]:
            plt.axvline(p, linestyle="--", alpha=0.3)
            plt.text(p, plt.ylim()[1]*0.8, f"{p}h", rotation=90, alpha=0.6)

    plt.tight_layout()
    plt.show()    

def plot_all_components_and_sum(t_hours, signal, freqs, fft_coeffs, selected_indices):
    n = len(signal)

    # Reconstruction de toutes les composantes sélectionnées
    components = []
    labels = []

    for k in selected_indices:
        comp = reconstruct_component(k, fft_coeffs, n)
        components.append(comp)

        if k == 0:
            labels.append("Moyenne")
        else:
            labels.append(f"{period_from_freq(freqs[k]):.2f} h")

    summed_signal = np.sum(components, axis=0)

    # ---- Figure 1 : toutes les composantes sur le même graphique ----
    plt.figure(figsize=(12, 5))
    for comp, lab in zip(components, labels):
        plt.plot(t_hours, comp, linewidth=1.5, label=lab)

    plt.title("Composantes fréquentielles reconstruites")
    plt.xlabel("Temps (h)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Composantes", ncol=2)
    plt.tight_layout()
    plt.show()

    # ---- Figure 2 : somme des composantes ----
    plt.figure(figsize=(12, 5))
    plt.plot(t_hours, summed_signal, linewidth=2, label="Somme des composantes")
    plt.plot(t_hours, signal, "--", linewidth=2, label="Signal original")
    plt.title("Signal reconstruit par addition des composantes")
    plt.xlabel("Temps (h)")
    plt.ylabel("Irradiance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_components_sum_and_original_same_axis(
    t_hours, signal, freqs, fft_coeffs, selected_indices
):
    n = len(signal)

    components = [reconstruct_component(k, fft_coeffs, n) for k in selected_indices]
    summed_signal = np.sum(components, axis=0)

    plt.figure(figsize=(12, 6))

    for k, comp in zip(selected_indices, components):
        if k == 0:
            label = "Moyenne"
        else:
            label = f"{period_from_freq(freqs[k]):.2f} h"
        plt.plot(t_hours, comp, linewidth=1.1, alpha=0.8, label=label)

    plt.plot(t_hours, summed_signal, linewidth=3, label="Somme des composantes")
    plt.plot(t_hours, signal, "--", linewidth=2.5, label="Signal original")

    plt.title("Composantes fréquentielles, somme et signal original")
    plt.xlabel("Temps (h)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    if len(selected_indices) <= 10:
        plt.legend(title="Composantes", ncol=2)
    #plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

    # petit diagnostic
    rmse = np.sqrt(np.mean((summed_signal - signal) ** 2))
    print(f"RMSE reconstruction = {rmse:.6f}")


if __name__ == "__main__":
    dt_seconds = 5 * 60

    # 1. signal
    t_hours, S = create_synthetic_irradiance()

    plt.figure(figsize=(10, 4))
    plt.plot(t_hours, S, linewidth=2)
    plt.title("Signal synthétique d'irradiance")
    plt.xlabel("Temps (h)")
    plt.ylabel("Irradiance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. FFT
    freqs, fft_coeffs = fft_analysis(S, dt_seconds)

    # 3. spectre
    plot_spectrum(freqs, fft_coeffs)

    # 3.1 spectre
    plot_loglog_spectrum(freqs, fft_coeffs)

    # 4. sélection des composantes
    selected_indices = select_main_components(freqs, fft_coeffs, n_oscillatory=5)

    print("Composantes affichées :")
    for k in selected_indices:
        if k == 0:
            print(f"  k={k}: moyenne")
        else:
            print(f"  k={k}: période = {period_from_freq(freqs[k]):.2f} h")

    # 5. figure pédagogique
    plot_fft_tutorial(t_hours, S, freqs, fft_coeffs, selected_indices)

    # 6. Affichage
    plot_all_components_and_sum(t_hours, S, freqs, fft_coeffs, selected_indices)

    # 7. Affichage sur le même graphique
    plot_components_sum_and_original_same_axis(t_hours, S, freqs, fft_coeffs, selected_indices)

    # 8. Affichage sur le même graphique
    selected_indices = list(range(len(fft_coeffs)))
    plot_components_sum_and_original_same_axis(t_hours, S, freqs, fft_coeffs, selected_indices)    