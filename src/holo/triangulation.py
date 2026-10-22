import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from .config import SAMPLE_RATE, LOWPASS_CUTOFF_HZ
from .ultrasonic import lowpass_for_v2, UltrasonicSonarGate

_SONAR_HELPER = UltrasonicSonarGate(sr=SAMPLE_RATE)

def align_and_slice_waveform(waveform, sr=SAMPLE_RATE, pre_ms=8.0, post_ms=35.0):
    raw = np.asarray(waveform, dtype=np.float32).flatten()
    lf = lowpass_for_v2(raw, sr=sr, cutoff_hz=LOWPASS_CUTOFF_HZ)
    peak_idx = int(np.argmax(np.abs(lf)))
    pre_samples = max(1, int((pre_ms / 1000.0) * sr))
    post_samples = max(1, int((post_ms / 1000.0) * sr))
    target_len = pre_samples + post_samples
    start = max(0, peak_idx - pre_samples)
    end = min(len(lf), peak_idx + post_samples)
    sliced = lf[start:end]
    if len(sliced) < target_len:
        pad_left = max(0, pre_samples - peak_idx)
        pad_right = max(0, target_len - len(sliced) - pad_left)
        sliced = np.pad(sliced, (pad_left, pad_right), mode="constant")
    noise_end = max(16, peak_idx - pre_samples)
    pre_noise = lf[:noise_end] if noise_end <= len(lf) else lf[:16]
    return sliced.astype(np.float32), pre_noise.astype(np.float32)

def extract_fourier_triangulation_vector(waveform, sr=SAMPLE_RATE, n_bins=48):
    raw = np.asarray(waveform, dtype=np.float32).flatten()
    aligned_lf, _ = align_and_slice_waveform(raw, sr=sr)
    n = len(aligned_lf)
    windowed = aligned_lf * np.hanning(n)
    fft_complex = np.fft.rfft(windowed)
    mag = np.abs(fft_complex) + 1e-9
    phase = np.unwrap(np.angle(fft_complex))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    valid_mask = freqs <= LOWPASS_CUTOFF_HZ
    mag_valid = mag[valid_mask]
    phase_valid = phase[valid_mask]
    indices = np.array_split(np.arange(len(mag_valid)), n_bins)
    pooled_mag = np.array([np.mean(np.log1p(mag_valid[idx])) for idx in indices if len(idx) > 0], dtype=np.float32)
    if len(pooled_mag) < n_bins:
        pooled_mag = np.pad(pooled_mag, (0, n_bins - len(pooled_mag)))
    pooled_mag = pooled_mag / (np.linalg.norm(pooled_mag) + 1e-9)
    early_len = max(8, n // 4)
    early_spec = np.abs(np.fft.rfft(aligned_lf[:early_len] * np.hanning(early_len)))
    early_freqs = np.fft.rfftfreq(early_len, d=1.0 / sr)
    hf_early = float(np.mean(early_spec[(early_freqs >= 2500) & (early_freqs <= 12000)] + 1e-9))
    lf_early = float(np.mean(early_spec[(early_freqs >= 150) & (early_freqs < 2500)] + 1e-9))
    dispersion_ratio = np.log(hf_early / lf_early)
    d_phase = np.diff(phase_valid) if len(phase_valid) > 2 else np.zeros(1, dtype=np.float32)
    phase_stats = np.array([float(np.mean(d_phase)), float(np.std(d_phase)), float(dispersion_ratio)], dtype=np.float32)
    ultra_vec = _SONAR_HELPER.extract_ultrasonic_feature_vector(raw)
    return np.concatenate([pooled_mag, phase_stats, ultra_vec]).astype(np.float32)

class FourierTriangulator:
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
        self.scaler = StandardScaler()
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.classes_ = None
        self.templates_ = {}

    def fit(self, waveforms, y):
        X_four = np.vstack([extract_fourier_triangulation_vector(w, sr=self.sr) for w in waveforms])
        self.classes_ = np.unique(y)
        X_scaled = self.scaler.fit_transform(X_four)
        self.lda.fit(X_scaled, y)
        for cls in self.classes_:
            cls_vecs = X_scaled[y == cls]
            mean_vec = np.mean(cls_vecs, axis=0)
            self.templates_[cls] = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)
        return self

    def predict_proba(self, waveforms):
        if isinstance(waveforms, np.ndarray) and waveforms.ndim == 1:
            waveforms = [waveforms]
        X_four = np.vstack([extract_fourier_triangulation_vector(w, sr=self.sr) for w in waveforms])
        X_scaled = self.scaler.transform(X_four)
        lda_probs = self.lda.predict_proba(X_scaled)
        sim_matrix = np.zeros((len(X_scaled), len(self.classes_)), dtype=np.float64)
        for j, cls in enumerate(self.classes_):
            norms = np.linalg.norm(X_scaled, axis=1) + 1e-9
            cos_sim = np.dot(X_scaled, self.templates_[cls]) / norms
            sim_matrix[:, j] = np.exp(3.0 * cos_sim)
        sim_probs = sim_matrix / (np.sum(sim_matrix, axis=1, keepdims=True) + 1e-9)
        return 0.70 * lda_probs + 0.30 * sim_probs
