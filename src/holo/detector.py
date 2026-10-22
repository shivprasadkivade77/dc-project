import queue
from collections import deque
import numpy as np
from .config import PRE_BLOCKS, POST_BLOCKS, COOLDOWN_BLOCKS, RMS_THRESHOLD, DYNAMIC_MULTIPLIER, SAMPLE_RATE
from .ultrasonic import lowpass_for_v2

class ImpulseDetector:
    def __init__(self, threshold=RMS_THRESHOLD, dynamic_multiplier=DYNAMIC_MULTIPLIER, sr=SAMPLE_RATE):
        self.threshold = threshold
        self.dynamic_multiplier = dynamic_multiplier
        self.sr = sr
        self.pre_buffer = deque(maxlen=PRE_BLOCKS)
        self.ambient_rms_history = deque(maxlen=40)
        self.calibration_buffer = deque(maxlen=40)
        self.capturing = False
        self.captured_blocks = []
        self.post_counter = 0
        self.cooldown = 0
        self.event_queue = queue.Queue()

    def process_block(self, raw_block):
        block = np.asarray(raw_block, dtype=np.float32).flatten()
        lf_block = lowpass_for_v2(block, sr=self.sr)
        rms = float(np.sqrt(np.mean(lf_block**2) + 1e-12))
        peak = float(np.max(np.abs(lf_block)) + 1e-12)
        crest = peak / rms

        if self.cooldown > 0:
            self.cooldown -= 1
            self.pre_buffer.append(block)
            return

        if self.capturing:
            self.captured_blocks.append(block)
            self.post_counter -= 1
            if self.post_counter <= 0:
                self.event_queue.put(np.concatenate(self.captured_blocks).astype(np.float32))
                self.capturing = False
                self.captured_blocks = []
                self.cooldown = COOLDOWN_BLOCKS
            return

        ambient_floor = float(np.median(self.ambient_rms_history)) if self.ambient_rms_history else 0.002
        dynamic_thresh = max(self.threshold, ambient_floor * self.dynamic_multiplier)

        if rms > dynamic_thresh and crest > 1.85:
            self.capturing = True
            self.captured_blocks = list(self.pre_buffer) + [block]
            self.post_counter = POST_BLOCKS
        else:
            self.pre_buffer.append(block)
            self.ambient_rms_history.append(rms)
            self.calibration_buffer.append(block)

    def audio_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata
        self.process_block(mono.copy())
