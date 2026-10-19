import argparse
import numpy as np
import sounddevice as sd
from .config import SAMPLE_RATE, BLOCK_SIZE, RMS_THRESHOLD, DATA_DIR, CLASSES
from .detector import ImpulseDetector

def collect_data(label: str, num_samples: int, threshold: float):
    save_dir = DATA_DIR / label
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(save_dir.glob("*.npy")))

    detector = ImpulseDetector(threshold=threshold)
    print(f"\n[COLLECT] Recording {num_samples} samples for '{label}'...")
    print("Tap firmly on the target zone whenever ready. Press Ctrl+C to stop.\n")

    count = 0
    with sd.InputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, callback=detector.audio_callback
    ):
        while count < num_samples:
            waveform = detector.event_queue.get()
            filename = save_dir / f"{label}_{existing + count:04d}.npy"
            np.save(filename, waveform)
            count += 1
            peak = float(np.max(np.abs(waveform)))
            print(f"  Captured [{count}/{num_samples}] -> {filename.name} (Peak: {peak:.3f})")

    print(f"\nSaved {num_samples} samples inside {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect training taps")
    parser.add_argument("--label", choices=CLASSES, required=True, help="Target class label")
    parser.add_argument("--samples", type=int, default=50, help="Number of taps to record")
    parser.add_argument("--threshold", type=float, default=RMS_THRESHOLD, help="Mic RMS threshold")
    args = parser.parse_args()
    collect_data(args.label, args.samples, args.threshold)
