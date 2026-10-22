import time
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
from .config import SAMPLE_RATE, BLOCK_SIZE, DATA_DIR, CLASSES, ULTRASONIC_ENABLED
from .detector import ImpulseDetector
from .ultrasonic import UltrasonicSonarGate

def collect_samples(label, count=20):
    target_dir = DATA_DIR / label
    target_dir.mkdir(parents=True, exist_ok=True)
    detector = ImpulseDetector(sr=SAMPLE_RATE)
    sonar = UltrasonicSonarGate(sr=SAMPLE_RATE)
    if ULTRASONIC_ENABLED:
        sonar.start_emitter()
        time.sleep(0.4)
    print(f"Recording {count} samples for {label} at {SAMPLE_RATE} Hz...")
    recorded = 0
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE, callback=detector.audio_callback):
            while recorded < count:
                try:
                    waveform = detector.event_queue.get(timeout=0.2)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    out_path = target_dir / f"{label}_{ts}.wav"
                    sf.write(str(out_path), waveform, SAMPLE_RATE)
                    recorded += 1
                    print(f"  [{recorded}/{count}] Saved {out_path.name}")
                except Exception:
                    pass
    finally:
        sonar.stop_emitter()

def main():
    print("=== AcousticTap V3 Data Collector ===")
    for idx, cls in enumerate(CLASSES, 1):
        print(f"  {idx}. {cls}")
    choice = input("Select class number or name: ").strip()
    label = CLASSES[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(CLASSES) else choice
    n = input("Number of samples [default 25]: ").strip()
    collect_samples(label, int(n) if n.isdigit() else 25)

if __name__ == "__main__":
    main()
