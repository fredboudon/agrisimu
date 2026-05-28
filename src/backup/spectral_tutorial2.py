import numpy as np
import matplotlib.pyplot as plt


def create_signals(n=256):
    x = np.arange(n)

    # 1) constant
    s1 = np.full(n, 0.5)

    # 2) three plateaus, widths 1:2:1
    s2 = np.zeros(n)
    w = n // 4
    s2[:w] = 0.3
    s2[w:3*w] = 0.7
    s2[3*w:] = 0.3

    # 3) constant + one central spike
    s3 = np.full(n, 0.2)
    s3[n // 2] = 0.7

    return x, [s1, s2, s3]


def compute_spectrum(signal, dt=1.0):
    fft_coeffs = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=dt)
    amp = np.abs(fft_coeffs)

    a0 = amp[0]
    freqs_pos = freqs[1:]
    amp_pos = amp[1:]

    return a0, freqs_pos, amp_pos


def plot_composite_fft_periods(n=256, dt=1.0):
    x, signals = create_signals(n=n)

    titles = [
        "Constant (0.5)",
        "Three plateaus (0.3 → 0.7 → 0.3)",
        "Single central spike"
    ]

    fig = plt.figure(figsize=(16, 8))
    outer = fig.add_gridspec(2, 3, height_ratios=[1, 1.3], hspace=0.35, wspace=0.30)

    top_axes = [fig.add_subplot(outer[0, i]) for i in range(3)]
    dc_axes = []
    spec_axes = []

    for i in range(3):
        sub = outer[1, i].subgridspec(1, 2, width_ratios=[1.1, 4.9], wspace=0.18)
        dc_axes.append(fig.add_subplot(sub[0, 0]))
        spec_axes.append(fig.add_subplot(sub[0, 1]))

    for i, (signal, title) in enumerate(zip(signals, titles)):
        # --------------------
        # Top: signal
        # --------------------
        ax = top_axes[i]
        ax.step(x, signal, where="mid", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # --------------------
        # Bottom: spectrum
        # --------------------
        a0, freqs_pos, amp_pos = compute_spectrum(signal, dt=dt)

        # DC panel
        ax_dc = dc_axes[i]
        ax_dc.bar([0], [a0], width=0.6)
        ax_dc.set_title("Mean\n(f = 0)", fontsize=10)
        ax_dc.set_xticks([0])
        ax_dc.set_xticklabels(["DC"])
        ax_dc.grid(True, axis="y", alpha=0.3)
        ax_dc.set_ylim(0, a0 * 1.15 if a0 > 0 else 1)

        if i == 0:
            ax_dc.set_ylabel("Amplitude")

        # Period panel
        ax_sp = spec_axes[i]

        if np.max(amp_pos) > 0:
            amp_pos_norm = amp_pos / np.max(amp_pos)
        else:
            amp_pos_norm = amp_pos.copy()

        mask = amp_pos_norm > 0
        freqs_plot = freqs_pos[mask]
        amp_plot = amp_pos_norm[mask]

        if len(freqs_plot) > 0:
            periods = 1 / freqs_plot

            ax_sp.plot(periods, amp_plot, linewidth=2)
            ax_sp.set_xscale("log")
            ax_sp.invert_xaxis()
            ax_sp.set_ylim(0, 1.05)

            pmin = periods.min()
            pmax = periods.max()
            ax_sp.set_xlim(pmax, pmin)

            # ticks en périodes simples
            candidate_ticks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=float)
            ticks = candidate_ticks[(candidate_ticks >= pmin) & (candidate_ticks <= pmax)]
            if len(ticks) > 0:
                ax_sp.set_xticks(ticks)
                ax_sp.set_xticklabels([f"{int(t)}" for t in ticks], rotation=45)

        else:
            ax_sp.text(
                0.5, 0.5,
                "No non-zero\nfrequencies",
                ha="center", va="center",
                transform=ax_sp.transAxes
            )
            ax_sp.set_xlim(1, n)
            ax_sp.set_ylim(0, 1)

        ax_sp.set_title("Non-zero frequencies\n(as periods)", fontsize=10)
        ax_sp.set_xlabel("Period (samples)")
        ax_sp.grid(True, which="both", alpha=0.3)

        if i == 0:
            ax_sp.set_ylabel("Normalized amplitude")

    fig.suptitle("Simple signals and their spectra", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_composite_fft_periods(n=32, dt=1.0)