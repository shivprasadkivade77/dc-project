import queue
from collections import deque
import numpy as np
import sounddevice as sd
from .config import CFG

def get_input_channels() -> int:
    ch_cfg = CFG["audio"].get("channels", "auto")
    if ch_cfg != "auto":
        return int(ch_cfg)
    try:
        dev = sd.query_devices(kind="input")
        return min(2, max(1, int(dev.get("max_input_channels", 1))))
    except Exception:
        return 1

class ImpulseDetector:
    """Captures fixed-length audio snippets around sharp transients using config.yaml thresholds."""
    def __init__(self, threshold: float | None = None):
        det_cfg = CFG["detector"]
        aud_cfg = CFG["audio"]
        self.threshold = float(threshold if threshold is not None else det_cfg["rms_threshold"])
        self.peak_mult = float(det_cfg.get("peak_multiplier", 1.8))
        self.bg_snr_ratio = float(det_cfg.get("bg_snr_ratio", 2.3))
        self.ema_alpha = float(det_cfg.get("bg_ema_alpha", 0.95))

        self.pre_blocks = int(aud_cfg["pre_blocks"])
        self.post_blocks = int(aud_cfg["post_blocks"])
        self.cooldown_blocks = int(aud_cfg["cooldown_blocks"])

        self.pre_buffer = deque(maxlen=self.pre_blocks)
        self.capturing = False
        self.captured_blocks = []
        self.cooldown = 0
        self.bg_rms = 0.002
        self.event_queue = queue.Queue()

    def audio_callback(self, indata, frames, time_info, status):
        block = indata.copy()
        mono_block = block[:, 0] if block.ndim == 2 else block
        rms = float(np.sqrt(np.mean(mono_block ** 2)))
        peak = float(np.max(np.abs(mono_block)))

        if self.cooldown > 0:
            self.cooldown -= 1
            self.pre_buffer.append(block)
            # Do NOT update bg_rms during cooldown to avoid absorbing chassis reverberation
            return

        if self.capturing:
            self.captured_blocks.append(block)
            if len(self.captured_blocks) >= (self.pre_blocks + 1 + self.post_blocks):
                full_waveform = np.concatenate(self.captured_blocks, axis=0)
                if full_waveform.ndim == 2 and full_waveform.shape[1] == 1:
                    full_waveform = full_waveform[:, 0]
                self.event_queue.put(full_waveform)
                self.capturing = False
                self.captured_blocks = []
                self.cooldown = self.cooldown_blocks
        else:
            dynamic_thresh = max(self.threshold, self.bg_rms * self.bg_snr_ratio)
            if rms > dynamic_thresh and peak > (self.threshold * self.peak_mult):
                self.capturing = True
                pad_blocks = [np.zeros_like(block) for _ in range(self.pre_blocks - len(self.pre_buffer))]
                self.captured_blocks = pad_blocks + list(self.pre_buffer) + [block]
            else:
                self.pre_buffer.append(block)
                self.bg_rms = self.ema_alpha * self.bg_rms + (1.0 - self.ema_alpha) * rms
