import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from .config import CFG, SAMPLE_RATE

def make_onset_window(length: int, peak_offset: int = 32) -> np.ndarray:
    """
    Creates an asymmetric onset-preserving window.
    - [0 : peak_offset]: Smooth rising cosine taper from 0 to 1, reaching 1.0 right at peak_offset.
    - [peak_offset : length]: Smooth decaying cosine taper from 1 down to 0 at the window end.
    This guarantees that the transient direct-wave impact spike at peak_offset receives
    full 1.0 gain (100% transmission), while preventing boundary truncation artifacts.
    """
    w = np.ones(length, dtype=np.float64)
    if peak_offset > 0:
        rise = 0.5 * (1.0 - np.cos(np.pi * np.arange(peak_offset) / max(1, peak_offset)))
        w[:peak_offset] = rise
    tail_len = length - peak_offset
    if tail_len > 0:
        fall = 0.5 * (1.0 + np.cos(np.pi * np.arange(tail_len) / max(1, tail_len)))
        w[peak_offset:] = fall
    return w

def align_and_slice_waveform(waveform: np.ndarray):
    win_cfg = CFG["signal_window"]
    s_start_cfg = int(win_cfg["search_start_samples"])
    s_end_margin = int(win_cfg["search_end_margin"])
    pre_guard = int(win_cfg["pre_noise_guard_samples"])
    pre_peak = int(win_cfg["onset_pre_peak_samples"])
    onset_len = int(win_cfg["onset_length_samples"])
    wake_len = int(win_cfg["wake_length_samples"])

    mono = np.mean(waveform, axis=1) if waveform.ndim == 2 else waveform
    mono = np.asarray(mono, dtype=np.float64)

    search_start = min(s_start_cfg, len(mono) // 6)
    search_end = max(search_start + 512, len(mono) - s_end_margin)
    peak_idx = search_start + int(np.argmax(np.abs(mono[search_start:search_end])))

    # Ensure pre_noise strictly never includes or overlaps the peak transient
    if peak_idx > pre_guard + 32:
        noise_end = min(256, peak_idx - pre_guard)
    else:
        noise_end = max(16, peak_idx // 2) if peak_idx > 16 else max(1, peak_idx)
    pre_noise = mono[:noise_end]
    if len(pre_noise) == 0:
        pre_noise = np.array([1e-6], dtype=np.float64)

    onset_start = max(0, peak_idx - pre_peak)
    onset = mono[onset_start : onset_start + onset_len]
    if len(onset) < onset_len:
        onset = np.pad(onset, (0, onset_len - len(onset)))

    wake_start = max(0, peak_idx - pre_peak)
    wake = mono[wake_start : wake_start + wake_len]
    if len(wake) < wake_len:
        wake = np.pad(wake, (0, wake_len - len(wake)))

    return mono, peak_idx, pre_noise, onset, wake

def extract_fourier_triangulation_vector(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    ft_cfg = CFG["fourier_triangulation"]
    win_cfg = CFG["signal_window"]
    pre_peak = int(win_cfg.get("onset_pre_peak_samples", 32))
    num_bands = int(ft_cfg["num_bands"])
    f_min = float(ft_cfg["freq_min_hz"])
    f_max = float(ft_cfg["freq_max_hz"])
    hf_low, hf_high = float(ft_cfg["hf_band_hz"][0]), float(ft_cfg["hf_band_hz"][1])
    lf_low, lf_high = float(ft_cfg["lf_band_hz"][0]), float(ft_cfg["lf_band_hz"][1])
    max_lag = int(ft_cfg["gcc_max_lag_samples"])

    mono, peak_idx, pre_noise, onset, wake = align_and_slice_waveform(waveform)

    peak_amp = np.max(np.abs(wake)) + 1e-8
    peak_offset = min(pre_peak, peak_idx)
    onset_window = make_onset_window(len(onset), peak_offset=peak_offset)
    wake_window = make_onset_window(len(wake), peak_offset=peak_offset)

    norm_onset = (onset / peak_amp) * onset_window
    norm_wake = (wake / peak_amp) * wake_window

    fft_onset = np.fft.rfft(norm_onset)
    mag_onset = np.abs(fft_onset) + 1e-8
    freqs_onset = np.fft.rfftfreq(len(norm_onset), d=1.0 / sr)

    band_edges = np.geomspace(f_min, f_max, num=num_bands + 1)
    onset_band_energies = []
    for i in range(num_bands):
        mask = (freqs_onset >= band_edges[i]) & (freqs_onset < band_edges[i + 1])
        onset_band_energies.append(np.log10(np.mean(mag_onset[mask] ** 2) + 1e-10) if np.any(mask) else -10.0)

    fft_wake = np.fft.rfft(norm_wake)
    mag_wake = np.abs(fft_wake) + 1e-8
    freqs_wake = np.fft.rfftfreq(len(norm_wake), d=1.0 / sr)

    wake_band_energies = []
    for i in range(num_bands):
        mask = (freqs_wake >= band_edges[i]) & (freqs_wake < band_edges[i + 1])
        wake_band_energies.append(np.log10(np.mean(mag_wake[mask] ** 2) + 1e-10) if np.any(mask) else -10.0)

    # Sub-band ratios between neighboring resonance bands (captures spatial node shifts)
    band_diffs = list(np.diff(wake_band_energies))

    # Lamb-Wave Dispersion Delay
    hf_fft = np.where((freqs_wake >= hf_low) & (freqs_wake <= hf_high), fft_wake, 0.0)
    lf_fft = np.where((freqs_wake >= lf_low) & (freqs_wake <= lf_high), fft_wake, 0.0)
    hf_time_env = np.abs(np.fft.irfft(hf_fft))
    lf_time_env = np.abs(np.fft.irfft(lf_fft))

    hf_arrival = int(np.argmax(hf_time_env[:256]))
    lf_arrival = int(np.argmax(lf_time_env[:256]))
    dispersion_lag_ms = ((lf_arrival - hf_arrival) / sr) * 1000.0
    hf_to_lf_ratio = float(np.log10((np.sum(hf_time_env ** 2) + 1e-8) / (np.sum(lf_time_env ** 2) + 1e-8)))

    unwrapped_phase = np.unwrap(np.angle(fft_onset))
    group_delay = -np.diff(unwrapped_phase)
    gd_low = float(np.mean(group_delay[1:8]))
    gd_mid = float(np.mean(group_delay[8:32]))
    gd_high = float(np.mean(group_delay[32:min(90, len(group_delay))]))

    snr_db = float(10.0 * np.log10((np.mean(onset ** 2) + 1e-10) / (np.mean(pre_noise ** 2) + 1e-10)))

    if waveform.ndim == 2 and waveform.shape[1] >= 2:
        start_idx = max(0, peak_idx - 32)
        ch_l = waveform[start_idx : start_idx + 512, 0]
        ch_r = waveform[start_idx : start_idx + 512, 1]
        if len(ch_l) < 512:
            ch_l = np.pad(ch_l, (0, 512 - len(ch_l)))
            ch_r = np.pad(ch_r, (0, 512 - len(ch_r)))
        F_l = np.fft.rfft(ch_l, n=1024)
        F_r = np.fft.rfft(ch_r, n=1024)
        cross_spec = F_l * np.conj(F_r)
        gcc_phat = np.fft.irfft(cross_spec / (np.abs(cross_spec) + 1e-8), n=1024)
        gcc_shifted = np.fft.fftshift(gcc_phat)
        center = len(gcc_shifted) // 2
        lag_samples = int(np.argmax(gcc_shifted[center - max_lag : center + max_lag + 1])) - max_lag
        ild_db = float(10.0 * np.log10((np.sum(ch_l ** 2) + 1e-8) / (np.sum(ch_r ** 2) + 1e-8)))
    else:
        lag_samples = 0.0
        ild_db = 0.0

    return np.array(
        onset_band_energies
        + wake_band_energies
        + band_diffs
        + [dispersion_lag_ms, hf_to_lf_ratio, gd_low, gd_mid, gd_high, snr_db, float(lag_samples), ild_db],
        dtype=np.float64
    )

class FourierTriangulator:
    def __init__(self):
        ft_cfg = CFG["fourier_triangulation"]
        self.lda_weight = float(ft_cfg.get("lda_weight", 0.65))
        self.cos_temp = float(ft_cfg.get("cosine_temperature", 2.5))
        self.scaler = StandardScaler()
        self.lda = LinearDiscriminantAnalysis(
            solver=ft_cfg.get("lda_solver", "lsqr"),
            shrinkage=ft_cfg.get("lda_shrinkage", "auto")
        )
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

        norm_x = X_scaled / (np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-8)
        norm_c = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.dot(norm_x, norm_c.T) * self.cos_temp
        exp_sim = np.exp(cos_sim - np.max(cos_sim, axis=1, keepdims=True))
        cos_probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

        return self.lda_weight * lda_probs + (1.0 - self.lda_weight) * cos_probs
