import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from .config import SAMPLE_RATE

def align_and_slice_waveform(waveform: np.ndarray):
    """
    Finds the exact microsecond impact peak and isolates:
    1. pre_noise: Background noise before the tap
    2. onset: First 5.8 ms (256 samples) direct flexural wave before chassis edge reflections
    3. wake: 23.2 ms (1024 samples) structural cavity resonance
    """
    mono = np.mean(waveform, axis=1) if waveform.ndim == 2 else waveform
    mono = np.asarray(mono, dtype=np.float64)

    # Search for impact peak outside the very edges of the buffer
    search_start = min(256, len(mono) // 6)
    search_end = max(search_start + 512, len(mono) - 1024)
    peak_idx = search_start + int(np.argmax(np.abs(mono[search_start:search_end])))

    # 1. Background noise floor prior to impact
    noise_end = max(128, peak_idx - 96)
    pre_noise = mono[:noise_end]

    # 2. Direct ballistic wave onset (256 samples ≈ 5.8 ms around peak)
    onset_start = max(0, peak_idx - 32)
    onset = mono[onset_start : onset_start + 256]
    if len(onset) < 256:
        onset = np.pad(onset, (0, 256 - len(onset)))

    # 3. Chassis resonant wake (1024 samples ≈ 23.2 ms starting at impact)
    wake_start = max(0, peak_idx - 32)
    wake = mono[wake_start : wake_start + 1024]
    if len(wake) < 1024:
        wake = np.pad(wake, (0, 1024 - len(wake)))

    return mono, peak_idx, pre_noise, onset, wake

def extract_fourier_triangulation_vector(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Computes Fourier Transform Triangulation features:
    - Direct-wave 256-pt FFT log-power spectrum (captures initial direction prior to reflections)
    - Resonant-wake 1024-pt FFT sub-band ratios (captures Left vs Right internal cavity resonance)
    - Lamb-Wave Dispersion Delay (time lag between high-freq >3kHz FFT band and low-freq <900Hz band)
    - Spectral Phase Group Delay across frequency bands
    - Stereo GCC-PHAT (Generalized Cross-Correlation with Phase Transform) if 2 channels exist
    """
    mono, peak_idx, pre_noise, onset, wake = align_and_slice_waveform(waveform)

    # Normalize onset and wake by impact peak so triangulation is independent of tap force
    peak_amp = np.max(np.abs(wake)) + 1e-8
    norm_onset = (onset / peak_amp) * np.hanning(len(onset))
    norm_wake = (wake / peak_amp) * np.hanning(len(wake))

    # A. Direct-Wave Onset FFT (256-pt -> 129 bins, ~172 Hz resolution)
    fft_onset = np.fft.rfft(norm_onset)
    mag_onset = np.abs(fft_onset) + 1e-8
    freqs_onset = np.fft.rfftfreq(len(norm_onset), d=1.0 / sr)

    # Group onset FFT into 12 log-spaced spatial triangulation bands (150 Hz to 16 kHz)
    band_edges = np.geomspace(150, 16000, num=13)
    onset_band_energies = []
    for i in range(12):
        mask = (freqs_onset >= band_edges[i]) & (freqs_onset < band_edges[i + 1])
        onset_band_energies.append(np.log10(np.mean(mag_onset[mask] ** 2) + 1e-10) if np.any(mask) else -10.0)

    # B. Cavity Resonance Wake FFT (1024-pt -> 513 bins, ~43 Hz resolution)
    fft_wake = np.fft.rfft(norm_wake)
    mag_wake = np.abs(fft_wake) + 1e-8
    freqs_wake = np.fft.rfftfreq(len(norm_wake), d=1.0 / sr)

    wake_band_energies = []
    for i in range(12):
        mask = (freqs_wake >= band_edges[i]) & (freqs_wake < band_edges[i + 1])
        wake_band_energies.append(np.log10(np.mean(mag_wake[mask] ** 2) + 1e-10) if np.any(mask) else -10.0)

    # C. Lamb-Wave Dispersion Triangulation (High-Freq vs Low-Freq Arrival Time Lag via Inverse FFT)
    # In aluminum plates, high-frequency flexural waves travel faster than low-frequency waves.
    hf_fft = np.where((freqs_wake >= 3000) & (freqs_wake <= 14000), fft_wake, 0.0)
    lf_fft = np.where((freqs_wake >= 150) & (freqs_wake <= 900), fft_wake, 0.0)
    hf_time_env = np.abs(np.fft.irfft(hf_fft))
    lf_time_env = np.abs(np.fft.irfft(lf_fft))

    hf_arrival = int(np.argmax(hf_time_env[:256]))
    lf_arrival = int(np.argmax(lf_time_env[:256]))
    dispersion_lag_ms = ((lf_arrival - hf_arrival) / sr) * 1000.0
    hf_to_lf_ratio = float(np.log10((np.sum(hf_time_env ** 2) + 1e-8) / (np.sum(lf_time_env ** 2) + 1e-8)))

    # D. Fourier Phase Group Delay (-d(phi)/d(omega)) across Low, Mid, High bands
    unwrapped_phase = np.unwrap(np.angle(fft_onset))
    group_delay = -np.diff(unwrapped_phase)
    gd_low = float(np.mean(group_delay[1:8]))
    gd_mid = float(np.mean(group_delay[8:32]))
    gd_high = float(np.mean(group_delay[32:90]))

    # E. Pre-trigger SNR & Impulse Sharpness (helps Fourier Triangulator reject non-tap noise)
    snr_db = float(10.0 * np.log10(( np.mean(onset ** 2) + 1e-10 ) / ( np.mean(pre_noise ** 2) + 1e-10 )))

    # F. Stereo GCC-PHAT Cross-Correlation Lag & Inter-Channel Ratio (if 2 channels available)
    if waveform.ndim == 2 and waveform.shape[1] >= 2:
        ch_l = waveform[ max(0, peak_idx - 32) : max(0, peak_idx - 32) + 512, 0 ]
        ch_r = waveform[ max(0, peak_idx - 32) : max(0, peak_idx - 32) + 512, 1 ]
        F_l = np.fft.rfft(ch_l, n=1024)
        F_r = np.fft.rfft(ch_r, n=1024)
        cross_spec = F_l * np.conj(F_r)
        gcc_phat = np.fft.irfft(cross_spec / (np.abs(cross_spec) + 1e-8), n=1024)
        # Shift zero-lag to center and search within +/- 20 samples (~0.45 ms max physical mic spacing delay)
        gcc_shifted = np.fft.fftshift(gcc_phat)
        center = len(gcc_shifted) // 2
        lag_samples = int(np.argmax(gcc_shifted[center - 20 : center + 21])) - 20
        ild_db = float(10.0 * np.log10((np.sum(ch_l ** 2) + 1e-8) / (np.sum(ch_r ** 2) + 1e-8)))
    else:
        lag_samples = 0.0
        ild_db = 0.0

    return np.array(
        onset_band_energies
        + wake_band_energies
        + [dispersion_lag_ms, hf_to_lf_ratio, gd_low, gd_mid, gd_high, snr_db, float(lag_samples), ild_db],
        dtype=np.float64
    )

class FourierTriangulator:
    """
    Statistical Fourier Triangulation localizer using shrinkage-regularized
    Linear Discriminant projection + Spectral Template Cosine Similarity in FFT space.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.centroids = None

    def fit(self, X_fourier: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_fourier)
        self.lda.fit(X_scaled, y)
        classes = np.unique(y)
        self.centroids = np.array([np.mean(X_scaled[y == c], axis=0) for c in classes])
        return self

    def predict_proba(self, X_fourier: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_fourier)
        lda_probs = self.lda.predict_proba(X_scaled)

        # Combine LDA projection with spectral template cosine similarity (softmax temperature = 2.0)
        norm_x = X_scaled / (np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-8)
        norm_c = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.dot(norm_x, norm_c.T) * 2.5
        exp_sim = np.exp(cos_sim - np.max(cos_sim, axis=1, keepdims=True))
        cos_probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

        return 0.65 * lda_probs + 0.35 * cos_probs
