import numpy as np
import librosa
from .config import SAMPLE_RATE, WINDOW_PRE_PEAK, WINDOW_WAKE

def align_transient(waveform: np.ndarray, pre_peak: int = WINDOW_PRE_PEAK, length: int = WINDOW_WAKE) -> np.ndarray:
    """Extracts a peak-aligned active transient window, eliminating trailing silence."""
    peak_idx = int(np.argmax(np.abs(waveform)))
    start = peak_idx - pre_peak
    end = start + length

    if start >= 0 and end <= len(waveform):
        return waveform[start:end]

    pad_left = max(0, -start)
    pad_right = max(0, end - len(waveform))
    sliced = waveform[max(0, start):min(len(waveform), end)]
    return np.pad(sliced, (pad_left, pad_right), mode="constant")

def extract_features(waveform: np.ndarray, sr: int = SAMPLE_RATE, feature_set: str = "enhanced") -> np.ndarray:
    """Normalizes the peak-aligned tap waveform and extracts a scale-invariant 67-dimensional feature vector."""
    active_wave = align_transient(waveform)
    max_val = np.max(np.abs(active_wave))
    norm_wave = active_wave / (max_val + 1e-8)

    n_fft = 512
    hop_length = 128

    mfccs = librosa.feature.mfcc(
        y=norm_wave, sr=sr, n_mfcc=20, n_mels=40, n_fft=n_fft, hop_length=hop_length
    )
    features = [np.mean(mfccs, axis=1), np.std(mfccs, axis=1)]

    if feature_set == "enhanced":
        delta_mfcc = librosa.feature.delta(mfccs, width=3)
        centroid = librosa.feature.spectral_centroid(
            y=norm_wave, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=norm_wave, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        zcr = librosa.feature.zero_crossing_rate(
            y=norm_wave, frame_length=n_fft, hop_length=hop_length
        )
        # Scale-invariant normalized energy (measures impulse concentration, invariant to strike loudness)
        log_energy = np.array([np.log10(np.mean(norm_wave ** 2) + 1e-8)])

        features.extend([
            np.mean(delta_mfcc, axis=1),
            np.mean(centroid, axis=1), np.std(centroid, axis=1),
            np.mean(rolloff, axis=1), np.std(rolloff, axis=1),
            np.mean(zcr, axis=1), np.std(zcr, axis=1),
            log_energy
        ])

    return np.concatenate(features)
