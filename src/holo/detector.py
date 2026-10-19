import queue
from collections import deque
import numpy as np
import sounddevice as sd
from .config import PRE_BLOCKS, POST_BLOCKS, COOLDOWN_BLOCKS, RMS_THRESHOLD

def get_input_channels() -> int:
    """Detects whether the default macOS mic supports 2-channel stereo for GCC-PHAT or 1-channel mono."""
    try:
        dev = sd.query_devices(kind="input")
        return min(2, max(1, int(dev.get("max_input_channels", 1))))
    except Exception:
        return 1

class ImpulseDetector:
    """Captures fixed-length audio snippets when a sharp transient rises above the background floor."""
    def __init__(self, threshold: float = RMS_THRESHOLD):
        self.threshold = threshold
        self.pre_buffer = deque(maxlen=PRE_BLOCKS)
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
            self.bg_rms = 0.95 * self.bg_rms + 0.05 * rms
            return

        if self.capturing:
            self.captured_blocks.append(block)
            if len(self.captured_blocks) >= (PRE_BLOCKS + 1 + POST_BLOCKS):
                full_waveform = np.concatenate(self.captured_blocks, axis=0)
                if full_waveform.ndim == 2 and full_waveform.shape[1] == 1:
                    full_waveform = full_waveform[:, 0]
                self.event_queue.put(full_waveform)
                self.capturing = False
                self.captured_blocks = []
                self.cooldown = COOLDOWN_BLOCKS
        else:
            # Require both absolute threshold AND a sharp transient jump over rolling background noise
            dynamic_thresh = max(self.threshold, self.bg_rms * 2.5)
            if rms > dynamic_thresh and peak > (self.threshold * 2.0):
                self.capturing = True
                self.captured_blocks = list(self.pre_buffer) + [block]
            else:
                self.pre_buffer.append(block)
                self.bg_rms = 0.95 * self.bg_rms + 0.05 * rms
