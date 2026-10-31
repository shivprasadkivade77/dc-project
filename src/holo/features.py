from typing import Optional
import numpy as np
import librosa
from .config import SAMPLE_RATE, load_config
from .triangulation import align_and_slice_waveform

_cfg = load_config()
_feat_cfg = _cfg.get("features", {})
N_FFT = _feat_cfg.get("ml_n_fft", 256)
HOP_LENGTH = _feat_cfg.get("ml_hop_length", 64)

def extract_features(waveform: np.ndarray, sr: int = SAMPLE_RATE, feature_set: Optional[str] = "enhanced") -> np.ndarray:
    """
    Extracts ML features strictly from the peak-aligned 23.2 ms impact window
    (eliminating the 80% background silence averaging problem).
    """
    mono, peak_idx, pre_noise, onset, wake = align_and_slice_waveform(waveform)

    max_val = np.max(np.abs(wake)) + 1e-8
    norm_wake = (wake / max_val).astype(np.float32)
    norm_onset = (onset / max_val).astype(np.float32)

    mfccs = librosa.feature.mfcc(
        y=norm_wake, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=40
    )
    onset_mfcc = np.mean(mfccs[:, :3], axis=1)
    decay_mfcc = np.mean(mfccs[:, 3:], axis=1)
    std_mfcc = np.std(mfccs, axis=1)

    features = [onset_mfcc, decay_mfcc, std_mfcc]

    if feature_set == "enhanced":
        centroid = librosa.feature.spectral_centroid(y=norm_wake, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        rolloff = librosa.feature.spectral_rolloff(y=norm_wake, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        flatness = librosa.feature.spectral_flatness(y=norm_wake, n_fft=N_FFT, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y=norm_wake, frame_length=N_FFT, hop_length=HOP_LENGTH)

        rms_norm_wake = float(np.sqrt(np.mean(norm_wake ** 2)) + 1e-8)
        crest_factor = float(1.0 / rms_norm_wake)
        decay_tail = norm_wake[256:] if len(norm_wake) > 256 else norm_wake
        attack_decay_ratio = float(np.log10((np.mean(norm_onset ** 2) + 1e-8) / (np.mean(decay_tail ** 2) + 1e-8)))

        rms_wake = float(np.sqrt(np.mean(wake ** 2)) + 1e-8)
        rms_pre = float(np.sqrt(np.mean(pre_noise ** 2)) + 1e-8)
        snr_db = float(np.log10((rms_wake / rms_pre) + 1e-8))

        features.extend([
            np.mean(centroid, axis=1), np.std(centroid, axis=1),
            np.mean(rolloff, axis=1), np.std(rolloff, axis=1),
            np.mean(flatness, axis=1),
            np.mean(zcr, axis=1), np.std(zcr, axis=1),
            np.array([snr_db, crest_factor, attack_decay_ratio], dtype=np.float32)
        ])

    return np.concatenate(features)
