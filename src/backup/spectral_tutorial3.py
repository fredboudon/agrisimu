import numpy as np
import matplotlib.pyplot as plt


def create_spike_signal(n=256, baseline=0.2, spike_value=0.7):
    """
    Signal constant avec un pic unique au milieu.
    """
    x = np.arange(n)
    s = np.full(n, baseline, dtype=float)
    s[n // 2] = spike_value
    return x, s


def fft_analysis(signal, dt=1.0):
    freqs = np.fft.rfftfreq(len(signal), d=dt)
    fft_coeffs = np.fft.rfft(signal)
    return freqs, fft_coeffs


def reconstruct_component(k, fft_coeffs, n):
    """
    Reconstruit la composante temporelle correspondant à l'indice k.
    """
    idx = np.arange(n)

    if k == 0:
        return np.full(n, np.real(fft_coeffs[0]) / n)

    if n % 2 == 0 and k == n // 2:
        return np.real(fft_coeffs[k] * np.exp(2j * np.pi * k * idx / n)) / n

    return 2 * np.real(fft_coeffs[k] * np.exp(2j * np.pi * k * idx / n)) / n


def reconstruct_from_selected_indices(fft_coeffs, selected_indices, n):
    """
    Somme des composantes sélectionnées.
    """
    rec = np.zeros(n)
    for k in selected_indices:
        rec += reconstruct_component(k, fft_coeffs, n)
    return rec


def select_top_components(fft_coeffs, n_components):
    """
    Sélectionne la moyenne + les n composantes non nulles les plus fortes.
    """
    magnitudes = np.abs(fft_coeffs)
    sorted_idx = np.argsort(magnitudes[1:])[::-1] + 1
    return [0] + list(sorted_idx[:n_components])


def plot_spike_reconstructions(n=256):
    x, signal = create_spike_signal(n=n)
    freqs, fft_coeffs = fft_analysis(signal)
    n_freqs_total = len(fft_coeffs)

    reconstruction_levels = [3, 10, 50]

    fig, axes = plt.subplots(len(reconstruction_levels) + 2, 1, figsize=(11, 10), sharex=True)

    # 1. signal original
    axes[0].step(x, signal, where="mid", linewidth=2)
    axes[0].set_title("Original signal: flat baseline + one central spike")
    axes[0].set_ylabel("Signal")
    axes[0].set_ylim(0, 0.8)
    axes[0].grid(True, alpha=0.3)

    # 2..4 reconstructions partielles
    for i, n_comp in enumerate(reconstruction_levels, start=1):
        selected = select_top_components(fft_coeffs, n_components=n_comp)
        rec = reconstruct_from_selected_indices(fft_coeffs, selected, n)

        axes[i].step(x, signal, where="mid", linewidth=1.5, linestyle="--", label="Original")
        axes[i].plot(x, rec, linewidth=2, label=f"Reconstruction with {n_comp} frequencies")
        axes[i].set_ylabel("Signal")
        axes[i].set_ylim(0, 0.8)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right")

    # dernière : reconstruction complète
    selected_all = list(range(n_freqs_total))
    rec_all = reconstruct_from_selected_indices(fft_coeffs, selected_all, n)

    axes[-1].step(x, signal, where="mid", linewidth=1.5, linestyle="--", label="Original")
    axes[-1].plot(x, rec_all, linewidth=2, label="Reconstruction with all frequencies")
    axes[-1].set_title("Exact reconstruction when all frequencies are used")
    axes[-1].set_xlabel("Sample")
    axes[-1].set_ylabel("Signal")
    axes[-1].set_ylim(0, 0.8)
    axes[-1].grid(True, alpha=0.3)
    axes[-1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_spike_reconstructions(n=256)