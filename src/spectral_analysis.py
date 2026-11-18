import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d


def spectral_analysis_matrix(S, dt=5*60):
    """
    S : array (n_squares, n_times)
    """
    
    # Retirer la moyenne temporelle pour chaque carré
    S_centered = S - S.mean(axis=1, keepdims=True)
    
    N = S.shape[1]
    
    # FFT le long du temps
    fft_vals = np.fft.rfft(S_centered, axis=1)
    freqs = np.fft.rfftfreq(N, d=dt)
    
    power = np.abs(fft_vals)**2 / N
    
    return freqs, power

def spectral_intermittence_index(freqs, power, freq_threshold=1/(6*3600)):
    
    total_energy = power.sum(axis=1)
    high_freq_energy = power[:, freqs > freq_threshold].sum(axis=1)
    
    spectral_index = high_freq_energy / total_energy
    
    return spectral_index


def detect_spectral_bands(
    freqs,
    powers,
    smoothing_sigma=2,
    background_sigma=30,
    prominence=0.01,
    max_peaks=5
):
    """
    Détection robuste des bandes spectrales structurelles
    adaptée aux spectres fortement décroissants (type 1/f).
    Cette version est robuste aux valeurs nulles (ex. nuit).
    """

    # --- Spectre moyen ---
    mean_power = powers.mean(axis=0)

    # --- Lissage léger ---
    smoothed = gaussian_filter1d(mean_power, sigma=smoothing_sigma)

    # --- Estimation du fond spectral (très lissé) ---
    background = gaussian_filter1d(smoothed, sigma=background_sigma)

    # Eviter division par zéro
    eps = 1e-12
    background = np.where(background <= 0, eps, background)

    # --- Spectre aplati ---
    flattened = smoothed / background

    # --- Détection des pics sur le spectre aplati ---
    peaks, properties = find_peaks(
        flattened,
        prominence=prominence,
        distance=1
    )

    if len(peaks) == 0:
        return None

    # --- Largeur des pics ---
    widths, height, left_ips, right_ips = peak_widths(
        flattened, peaks, rel_height=0.9
    )

    bands = []
    band_energies = []

    for left, right in zip(left_ips, right_ips):
        i_min = int(np.floor(left))
        i_max = int(np.ceil(right))
        bands.append((i_min, i_max))

        # Energie réelle (spectre non aplati)
        band_energy = powers[:, i_min:i_max].sum(axis=1)
        band_energies.append(band_energy)

    band_energies = np.array(band_energies)

    # --- Importance énergétique réelle ---
    mean_band_energy = band_energies.mean(axis=1)
    idx_sorted = np.argsort(mean_band_energy)[::-1]

    idx_sorted = idx_sorted[:max_peaks]

    bands = [bands[i] for i in idx_sorted]
    peaks = peaks[idx_sorted]
    band_energies = band_energies[idx_sorted]
    mean_band_energy = mean_band_energy[idx_sorted]

    peak_freqs = freqs[peaks]
    periods = 1 / peak_freqs

    return {
        "peaks": peaks,
        "bands": bands,
        "periods": periods,
        "band_energies": band_energies,
        "mean_band_energy": mean_band_energy,
        "smoothed_spectrum": smoothed,
        "background": background,
        "flattened_spectrum": flattened
    }

def compute_bands_energy(powers, bands):
    mean_spectrum = powers.mean(axis=0)
    band_energies = []
    for (i_min, i_max) in bands:
        band_energy = mean_spectrum[i_min:i_max].sum()
        band_energies.append(band_energy)
    print(band_energies)
    return np.array(band_energies)
    

def time_axis(ax):
    ticks = [
        24*90,  # 3 mois
        24*30,  # 1 mois
        24*7,   # 1 semaine
        24,     # 1 jour
        12,
        6,
        1,      # 1 heure
        0.5,
        1/6,    # 10 min
    ]
    
    labels = [
        "3 m",
        "1 m",
        "1 w",
        "1 d",
        "12 h",
        "6 h",
        "1 h",
        "30 min",
        "10 min",
    ]
  
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

def plot_spectrum_in_period(freqs, power, square_index=0, log_scale = False):
    """
    Affiche le spectre en fonction des périodes (heures)
    
    freqs : array (n_freqs,)
    power : array (n_squares, n_freqs)
    square_index : indice du carré à afficher
    """
    
    # On enlève la fréquence 0 (sinon division par zéro)
    freqs_nonzero = freqs[1:]
    power_nonzero = power[square_index, 1:]
    
    # Conversion en périodes (heures)
    periods_hours = 1 / freqs_nonzero / 3600
    
    plt.figure(figsize=(8,5))
    plt.plot(periods_hours, power_nonzero)

    plt.axvline(24, linestyle="--", alpha=0.5)
    plt.axvline(12, linestyle="--", alpha=0.5)
    plt.axvline(1, linestyle="--", alpha=0.5)
    
    plt.xscale("log")  # très important pour lisibilité
    if log_scale : plt.yscale("log")
    plt.xlabel("Period")
    plt.ylabel("Spectral power")
    plt.title(f"Power spectrum of the shading rate of plot {square_index}")
    plt.gca().invert_xaxis()  # longues périodes à gauche (plus intuitif)
    time_axis(plt.gca())
    
    plt.grid(True)

def plot_spectral_bands_period(freqs, powers, results, 
                               time_unit="hours",
                               log_scale=False):
    """
    Affiche le spectre moyen en fonction de la période au lieu de la fréquence.

    Parameters
    ----------
    freqs : array (n_freq,)
    powers : array (n_squares, n_freq)
    results : dict retourné par detect_spectral_bands
    time_unit : str
        "minutes" ou "hours"
    log_scale : bool
    """
    print('R',results)
    mean_power = powers.mean(axis=0)

    # Eviter division par 0
    valid = freqs > 0
    freqs_valid = freqs[valid]
    power_valid = mean_power[valid]

    periods = 1 / freqs_valid 

    # Conversion optionnelle
    if time_unit == "minutes":
        periods = periods / 60
    elif time_unit == "hours":
        periods = periods / 3600


    # Trier par période croissante
    sort_idx = np.argsort(periods)
    periods = periods[sort_idx]
    power_valid = power_valid[sort_idx]

    plt.figure(figsize=(8,5))

    if log_scale:
        plt.loglog(periods, power_valid)
    else:
        plt.plot(periods, power_valid)

    # Pics
    peak_freqs = freqs[results['peaks']]
    peak_periods = 1 / peak_freqs
    if time_unit == "minutes":
        peak_periods = peak_periods / 60
    elif time_unit == "hours":
        peak_periods = peak_periods / 3600

    plt.scatter(peak_periods,
                mean_power[results['peaks']], color='red', label='Peaks')

    # Bandes
    for (i_min, i_max) in results['bands']:
        f_min = freqs[i_min]
        f_max = freqs[i_max]

        if f_min > 0 and f_max > 0:
            T_min = 1 / f_max
            T_max = 1 / f_min

            if time_unit == "minutes":
                T_min /= 60
                T_max /= 60
            elif time_unit == "hours":
                T_min /= 3600
                T_max /= 3600

            plt.axvspan(T_min, T_max, alpha=0.2)

    if not log_scale:
        plt.xscale("log")
    plt.xlabel(f"Period")
    plt.gca().invert_xaxis()  # longues périodes à gauche (plus intuitif)
    time_axis(plt.gca())
    plt.ylabel("Power")
    plt.title("Spectral peaks and integrated bands (period domain)")


def plot_all_spectra(freqs, power, log_scale = False):
    
    freqs_nonzero = freqs[1:]
    periods_hours = 1 / freqs_nonzero / 3600
    
    plt.figure(figsize=(8,5))
    
    for i in range(power.shape[0]):
        plt.plot(periods_hours, power[i,1:], alpha=0.3)

    plt.axvline(24, linestyle="--", alpha=0.5)
    plt.axvline(12, linestyle="--", alpha=0.5)
    plt.axvline(1, linestyle="--", alpha=0.5)
    
    plt.xscale("log")
    plt.gca().invert_xaxis()
    time_axis(plt.gca())
    if log_scale:
        plt.yscale("log")

    
    plt.xlabel("Period")
    plt.ylabel("Spectral power")
    plt.title("All spectra")
    plt.grid(True)

def plot_mean_spectrum(freqs, power, log_scale=False):

    freqs_nonzero = freqs[1:]
    periods_hours = 1 / freqs_nonzero / 3600

    power_nonzero = power[:, 1:]

    mean_power = np.mean(power_nonzero, axis=0)
    std_power = np.std(power_nonzero, axis=0)

    plt.figure(figsize=(8,5))

    plt.plot(periods_hours, mean_power, label="Mean")
    plt.fill_between(
        periods_hours,
        mean_power - std_power,
        mean_power + std_power,
        alpha=0.3,
        label="±1 standard deviation"
    )

    plt.xscale("log")
    plt.gca().invert_xaxis()
    time_axis(plt.gca())

    if log_scale:
        plt.yscale("log")

    plt.xlabel("Period")
    plt.ylabel("Spectral power")
    plt.title("Mean spectrum")

    plt.legend()
    plt.grid(True)

def plot_spectral_map(freqs, power, log_scale = True):
    
    freqs_nonzero = freqs[1:]
    periods_hours = 1 / freqs_nonzero / 3600
    
    power_nonzero = power[:,1:]
    
    # éviter log(0)
    if log_scale:
        power = np.log10(power_nonzero + 1e-12)
    
    plt.figure(figsize=(8,6))
    
    plt.imshow(power,
               aspect='auto',
               extent=[periods_hours.max(), periods_hours.min(), 0, power.shape[0]])
    
    plt.xscale("log")
    plt.xlabel("Period")
    plt.ylabel("Plot index")
    plt.title("Spectral map (log10)")
    plt.colorbar(label="log10(Spectral power)")
    time_axis(plt.gca())

def plot_mean_spectrum_with_bands(freqs, powers, results,
                            time_unit="hours",
                            log_scale=False):

    # --- Moyenne et écart-type ---
    mean_power = np.mean(powers, axis=0)
    std_power = np.std(powers, axis=0)

    # --- Éviter f=0 ---
    valid = freqs > 0
    freqs_valid = freqs[valid]
    mean_power = mean_power[valid]
    std_power = std_power[valid]

    # --- Conversion en période ---
    periods = 1 / freqs_valid

    if time_unit == "minutes":
        periods /= 60
    elif time_unit == "hours":
        periods /= 3600

    # --- Trier par période croissante ---
    sort_idx = np.argsort(periods)
    periods = periods[sort_idx]
    mean_power = mean_power[sort_idx]
    std_power = std_power[sort_idx]

    # --- Plot ---
    plt.figure(figsize=(8, 5))

    if log_scale:
        plt.yscale("log")

    plt.plot(periods, mean_power, label="Mean")
    plt.fill_between(periods,
                     mean_power - std_power,
                     mean_power + std_power,
                     alpha=0.3,
                     label="±1 standard deviation")

    # --- Pics ---
    peak_freqs = freqs[results['peaks']]
    peak_periods = 1 / peak_freqs

    if time_unit == "minutes":
        peak_periods /= 60
    elif time_unit == "hours":
        peak_periods /= 3600

    plt.scatter(peak_periods,
                np.mean(powers, axis=0)[results['peaks']],
                color='red', label='Peaks')

    # --- Bandes ---
    for (i_min, i_max) in results['bands']:
        f_min = freqs[i_min]
        f_max = freqs[i_max]

        if f_min > 0 and f_max > 0:
            T_min = 1 / f_max
            T_max = 1 / f_min

            if time_unit == "minutes":
                T_min /= 60
                T_max /= 60
            elif time_unit == "hours":
                T_min /= 3600
                T_max /= 3600

            plt.axvspan(T_min, T_max, alpha=0.2)

    # --- Axes ---
    plt.xscale("log")
    plt.gca().invert_xaxis()
    time_axis(plt.gca())

    plt.xlabel(f"Period")
    plt.ylabel("Spectral power")
    plt.title("Mean spectrum + spectral bands")

    plt.legend()
    plt.grid(True)

from math import floor
def plot_reference_band_energy(ref_bands, freqs, energy):
    plt.figure(figsize=(8, 4))
    plt.bar(ref_bands['peaks'], energy, width=[b[1]-b[0] for b in ref_bands['bands']])
    labels = [] 
    for peak in ref_bands['peaks']:
        period_hours = 1 / freqs[peak] / 3600
        period_minutes =  int(floor((period_hours % 1) * 60))
        period_hours = int(floor(period_hours))
        label = f"{period_hours}h{str(period_minutes).zfill(2)}"
        labels.append(label)
    plt.gca().invert_xaxis()  # longues périodes à gauche (plus intuitif)
    plt.xticks(ref_bands['peaks'], labels, rotation=45)
    plt.xlabel('Peak period')
    plt.ylabel('Energy')
    plt.title('Energy of reference spectral bands')    