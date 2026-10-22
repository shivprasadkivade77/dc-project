import numpy as np
import librosa
from .config import SAMPLE_RATE, LOWPASS_CUTOFF_HZ
from .triangulation import align_and_slice_waveform, extract_fourier_triangulation_vector
from .ultrasonic import lowpass_for_v2

def extract_features(waveform, sr=SAMPLE_RATE, feature_set="all"):
    raw = np.asarray(waveform, dtype=np.float32).flatten()
    aligned, pre_noise = align_and_slice_waveform(raw, sr=sr)
    rms = float(np.sqrt(np.mean(aligned**2) + 1e-12))
    peak = float(np.max(np.abs(aligned)) + 1e-12)
    crest_factor = peak / rms
    noise_rms = float(np.sqrt(np.mean(pre_noise**2) + 1e-12))
    snr_db = float(20.0 * np.log10(rms / noise_rms))
    
    # Normalize aligned waveform so light taps and hard taps share the same spectral/envelope scale
    norm_aligned = (aligned / peak).astype(np.float32)
    norm_rms = float(np.sqrt(np.mean(norm_aligned**2) + 1e-12))
    
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=norm_aligned, frame_length=min(256, len(norm_aligned)), hop_length=64)))
    split_pt = max(16, len(norm_aligned) // 3)
    attack_rms = float(np.sqrt(np.mean(norm_aligned[:split_pt] ** 2) + 1e-12))
    decay_rms = float(np.sqrt(np.mean(norm_aligned[split_pt:] ** 2) + 1e-12))
    attack_decay_ratio = np.log(attack_rms / decay_rms)
    
    temporal_feats = np.array([norm_rms, 1.0, crest_factor, np.clip(snr_db, 0.0, 25.0), zcr, attack_decay_ratio], dtype=np.float32)
    if feature_set == "temporal":
        return temporal_feats
        
    n_fft = min(512, 2 ** int(np.floor(np.log2(max(64, len(norm_aligned))))))
    hop = max(32, n_fft // 4)
    mfccs = librosa.feature.mfcc(y=norm_aligned, sr=sr, n_mfcc=13, n_mels=40, n_fft=n_fft, hop_length=hop, fmax=LOWPASS_CUTOFF_HZ)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=norm_aligned, sr=sr, n_fft=n_fft, hop_length=hop))) / sr
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=norm_aligned, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85))) / sr
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=norm_aligned, n_fft=n_fft, hop_length=hop)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=norm_aligned, sr=sr, n_fft=n_fft, hop_length=hop))) / sr
    spectral_feats = np.concatenate([mfcc_mean, mfcc_std, np.array([centroid, rolloff, flatness, bandwidth], dtype=np.float32)]).astype(np.float32)
    if feature_set == "spectral":
        return spectral_feats
    fourier_feats = extract_fourier_triangulation_vector(raw, sr=sr, n_bins=32)
    return np.concatenate([temporal_feats, spectral_feats, fourier_feats]).astype(np.float32)
