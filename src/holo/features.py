import numpy as np
import librosa
from .config import CFG, SAMPLE_RATE
from .triangulation import align_and_slice_waveform

def extract_features(waveform: np.ndarray, sr: int = SAMPLE_RATE, feature_set: str | None = None) -> np.ndarray:
    feat_cfg = CFG["features"]
    mode = feature_set if feature_set is not None else feat_cfg.get("mode", "enhanced")
    n_mfcc = int(feat_cfg.get("n_mfcc", 20))
    n_fft = int(feat_cfg.get("n_fft", 256))
    hop_length = int(feat_cfg.get("hop_length", 64))
    onset_frames = int(feat_cfg.get("onset_frames", 3))

    mono, peak_idx, pre_noise, onset, wake = align_and_slice_waveform(waveform)

    max_val = np.max(np.abs(wake)) + 1e-8
    norm_wake = (wake / max_val).astype(np.float32)

    mfccs = librosa.feature.mfcc(
        y=norm_wake, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    onset_mfcc = np.mean(mfccs[:, :onset_frames], axis=1)
    decay_mfcc = np.mean(mfccs[:, onset_frames:], axis=1)
    std_mfcc = np.std(mfccs, axis=1)

    features = [onset_mfcc, decay_mfcc, std_mfcc]

    if mode == "enhanced":
        centroid = librosa.feature.spectral_centroid(y=norm_wake, sr=sr, n_fft=n_fft, hop_length=hop_length)
        rolloff = librosa.feature.spectral_rolloff(y=norm_wake, sr=sr, n_fft=n_fft, hop_length=hop_length)
        flatness = librosa.feature.spectral_flatness(y=norm_wake, n_fft=n_fft, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y=norm_wake, frame_length=n_fft, hop_length=hop_length)

        rms_wake = float(np.sqrt(np.mean(wake ** 2)) + 1e-8)
        rms_pre = float(np.sqrt(np.mean(pre_noise ** 2)) + 1e-8)
        snr_db = np.log10((rms_wake / rms_pre) + 1e-8)
        crest_factor = float(max_val / rms_wake)
        attack_decay_ratio = float(np.log10((np.mean(onset ** 2) + 1e-8) / (np.mean(wake[len(onset):] ** 2) + 1e-8)))

        features.extend([
            np.mean(centroid, axis=1), np.std(centroid, axis=1),
            np.mean(rolloff, axis=1), np.std(rolloff, axis=1),
            np.mean(flatness, axis=1),
            np.mean(zcr, axis=1), np.std(zcr, axis=1),
            np.array([snr_db, crest_factor, attack_decay_ratio], dtype=np.float32)
        ])

    return np.concatenate(features)
