import numpy as np
import sounddevice as sd
from .config import (
    SAMPLE_RATE, LOWPASS_CUTOFF_HZ, LEFT_ULTRA_HZ, RIGHT_ULTRA_HZ,
    SPEAKER_AMPLITUDE, CARRIER_BW_HZ, DOPPLER_MIN_HZ, DOPPLER_MAX_HZ,
    ALPHA_DOPPLER, GATE_THRESHOLD, GATE_STEEPNESS
)

def lowpass_for_v2(waveform: np.ndarray, sr: int = SAMPLE_RATE, cutoff_hz: float = LOWPASS_CUTOFF_HZ) -> np.ndarray:
    y = np.asarray(waveform, dtype=np.float32).flatten()
    n = len(y)
    if n < 8 or sr <= cutoff_hz * 2.0:
        return y
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    transition_bw = 500.0
    mask = np.ones_like(freqs, dtype=np.float32)
    above = freqs >= (cutoff_hz + transition_bw)
    trans = (freqs > cutoff_hz) & (~above)
    mask[above] = 0.0
    mask[trans] = 0.5 * (1.0 + np.cos(np.pi * (freqs[trans] - cutoff_hz) / transition_bw))
    return np.fft.irfft(spec * mask, n=n).astype(np.float32)

class UltrasonicSonarGate:
    def __init__(self, sr=SAMPLE_RATE, left_freq=LEFT_ULTRA_HZ, right_freq=RIGHT_ULTRA_HZ,
                 amplitude=SPEAKER_AMPLITUDE, carrier_bw=CARRIER_BW_HZ, doppler_min=DOPPLER_MIN_HZ,
                 doppler_max=DOPPLER_MAX_HZ, alpha_doppler=ALPHA_DOPPLER, gate_threshold=GATE_THRESHOLD,
                 gate_steepness=GATE_STEEPNESS):
        self.sr = sr
        self.left_freq = left_freq
        self.right_freq = right_freq
        self.amplitude = amplitude
        self.carrier_bw = carrier_bw
        self.doppler_min = doppler_min
        self.doppler_max = doppler_max
        self.alpha = alpha_doppler
        self.threshold = gate_threshold
        self.steepness = gate_steepness
        self._phase_l = 0.0
        self._phase_r = 0.0
        self._out_stream = None
        self.base_mean = {"left": 1e-3, "right": 1e-3}
        self.base_std = {"left": 2e-4, "right": 2e-4}
        self.base_doppler_mean = {"left": 0.08, "right": 0.08}
        self.base_doppler_std = {"left": 0.02, "right": 0.02}

    def _speaker_callback(self, outdata, frames, time_info, status):
        t = np.arange(frames, dtype=np.float64) / self.sr
        outdata[:, 0] = (self.amplitude * np.sin(2.0 * np.pi * self.left_freq * t + self._phase_l)).astype(np.float32)
        outdata[:, 1] = (self.amplitude * np.sin(2.0 * np.pi * self.right_freq * t + self._phase_r)).astype(np.float32)
        self._phase_l = float((self._phase_l + 2.0 * np.pi * self.left_freq * frames / self.sr) % (2.0 * np.pi))
        self._phase_r = float((self._phase_r + 2.0 * np.pi * self.right_freq * frames / self.sr) % (2.0 * np.pi))

    def start_emitter(self):
        if self._out_stream is None:
            self._out_stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32", blocksize=2048, callback=self._speaker_callback)
            self._out_stream.start()

    def stop_emitter(self):
        if self._out_stream is not None:
            self._out_stream.stop()
            self._out_stream.close()
            self._out_stream = None

    def extract_band_metrics(self, waveform, carrier_freq):
        y = np.asarray(waveform, dtype=np.float32).flatten()
        n = len(y)
        if n < 64 or self.sr <= carrier_freq * 2.0:
            return 0.0, 0.0
        windowed = y * np.hanning(n)
        spec = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sr)
        dist = np.abs(freqs - carrier_freq)
        carrier_mask = dist <= self.carrier_bw
        sideband_mask = (dist >= self.doppler_min) & (dist <= self.doppler_max)
        if not np.any(carrier_mask) or not np.any(sideband_mask):
            return 0.0, 0.0
        carrier_energy = float(np.mean(spec[carrier_mask])) + 1e-9
        sideband_energy = float(np.mean(spec[sideband_mask]))
        return carrier_energy, float(sideband_energy / carrier_energy)

    def has_active_carrier(self, waveform):
        y = np.asarray(waveform, dtype=np.float32).flatten()
        if len(y) < 128 or self.sr < 44100:
            return False
        c_l, _ = self.extract_band_metrics(y, self.left_freq)
        c_r, _ = self.extract_band_metrics(y, self.right_freq)
        return (c_l + c_r) > 1e-4

    def calibrate_baseline(self, empty_windows):
        if not empty_windows:
            return
        for side, freq in [("left", self.left_freq), ("right", self.right_freq)]:
            c_vals, d_vals = [], []
            for w in empty_windows:
                c_en, d_rat = self.extract_band_metrics(w, freq)
                c_vals.append(c_en)
                d_vals.append(d_rat)
            self.base_mean[side] = float(np.mean(c_vals))
            self.base_std[side] = max(float(np.std(c_vals)), self.base_mean[side] * 0.08, 1e-6)
            self.base_doppler_mean[side] = float(np.mean(d_vals))
            self.base_doppler_std[side] = max(float(np.std(d_vals)), 0.015)

    def _side_gate(self, waveform, side, freq):
        c_en, d_rat = self.extract_band_metrics(waveform, freq)
        z_doppler = max(0.0, (d_rat - self.base_doppler_mean[side]) / self.base_doppler_std[side])
        z_reflect = abs(c_en - self.base_mean[side]) / self.base_std[side]
        score = float(self.alpha * z_doppler + (1.0 - self.alpha) * z_reflect)
        gate = float(1.0 / (1.0 + np.exp(-self.steepness * (score - self.threshold))))
        return gate, score

    def evaluate_window(self, raw_waveform):
        if not self.has_active_carrier(raw_waveform):
            return {"gate_left": 1.0, "gate_right": 1.0, "score_left": 0.0, "score_right": 0.0, "active": False}
        gate_l, score_l = self._side_gate(raw_waveform, "left", self.left_freq)
        gate_r, score_r = self._side_gate(raw_waveform, "right", self.right_freq)
        return {"gate_left": gate_l, "gate_right": gate_r, "score_left": score_l, "score_right": score_r, "active": True}

    def extract_ultrasonic_feature_vector(self, raw_waveform):
        if not self.has_active_carrier(raw_waveform):
            return np.zeros(6, dtype=np.float32)
        c_l, d_l = self.extract_band_metrics(raw_waveform, self.left_freq)
        c_r, d_r = self.extract_band_metrics(raw_waveform, self.right_freq)
        c_asym = (c_l - c_r) / (c_l + c_r + 1e-9)
        d_asym = (d_l - d_r) / (d_l + d_r + 1e-9)
        return np.array([np.log1p(c_l), np.log1p(c_r), d_l, d_r, c_asym, d_asym], dtype=np.float32)
