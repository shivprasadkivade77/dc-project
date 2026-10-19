import numpy as np
import librosa
from .config import SAMPLE_RATE

def extract_features(waveform: np.ndarray, sr: int = SAMPLE_RATE, feature_set: str = "enhanced") -> np.ndarray:
    """Normalizes the tap waveform and extracts a 67-dimensional acoustic feature vector."""
    max_val = np.max(np.abs(waveform))
    norm_wave = waveform / (max_val + 1e-8)

    n_fft = 512
    hop_length = 128

    mfccs = librosa.feature.mfcc(
        y=norm_wave, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length
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
        log_energy = np.array([np.log10(np.mean(waveform ** 2) + 1e-8)])

        features.extend([
            np.mean(delta_mfcc, axis=1),
            np.mean(centroid, axis=1), np.std(centroid, axis=1),
            np.mean(rolloff, axis=1), np.std(rolloff, axis=1),
            np.mean(zcr, axis=1), np.std(zcr, axis=1),
            log_energy
        ])

    return np.concatenate(features)
