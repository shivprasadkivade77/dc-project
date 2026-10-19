import queue
from collections import deque
import numpy as np
from .config import PRE_BLOCKS, POST_BLOCKS, COOLDOWN_BLOCKS, RMS_THRESHOLD

class ImpulseDetector:
    """Captures fixed-length audio snippets around sharp acoustic transients."""
    def __init__(self, threshold: float = RMS_THRESHOLD):
        self.threshold = threshold
        self.pre_buffer = deque(maxlen=PRE_BLOCKS)
        self.capturing = False
        self.captured_blocks = []
        self.cooldown = 0
        self.event_queue = queue.Queue()

    def audio_callback(self, indata, frames, time_info, status):
        mono_block = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(mono_block ** 2)))

        if self.cooldown > 0:
            self.cooldown -= 1
            self.pre_buffer.append(mono_block)
            return

        if self.capturing:
            self.captured_blocks.append(mono_block)
            if len(self.captured_blocks) >= (PRE_BLOCKS + 1 + POST_BLOCKS):
                full_waveform = np.concatenate(self.captured_blocks)
                self.event_queue.put(full_waveform)
                self.capturing = False
                self.captured_blocks = []
                self.cooldown = COOLDOWN_BLOCKS
        else:
            if rms > self.threshold:
                self.capturing = True
                self.captured_blocks = list(self.pre_buffer) + [mono_block]
            else:
                self.pre_buffer.append(mono_block)
