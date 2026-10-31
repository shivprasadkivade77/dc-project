import random
import argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
from .config import SAMPLE_RATE, BLOCK_SIZE, RMS_THRESHOLD, DATA_DIR, CLASSES
from .detector import ImpulseDetector, get_input_channels

def save_sample(save_dir: Path, label: str, waveform: np.ndarray) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(save_dir.glob("*.npy")))
    filename = save_dir / f"{label}_{existing:04d}.npy"
    np.save(filename, waveform)
    return filename

def collect_single_class(label: str, num_samples: int, threshold: float):
    save_dir = DATA_DIR / label
    channels = get_input_channels()
    detector = ImpulseDetector(threshold=threshold)
    print(f"\n[COLLECT - SINGLE CLASS] Recording {num_samples} samples for '{label}' (Channels: {channels})...")
    print("Tap firmly on the target zone whenever ready. Press Ctrl+C to stop.\n")

    count = 0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=channels, callback=detector.audio_callback
        ):
            while count < num_samples:
                waveform = detector.event_queue.get()
                saved_file = save_sample(save_dir, label, waveform)
                count += 1
                peak = float(np.max(np.abs(waveform)))
                print(f"  Captured [{count}/{num_samples}] -> {saved_file.name} (Peak: {peak:.3f})")
    except KeyboardInterrupt:
        print(f"\n[Interrupted] Saved {count} samples for '{label}'.")
        return

    print(f"\nSaved {num_samples} samples inside {save_dir}")

def collect_group_interleaved(samples_per_class: int, threshold: float):
    channels = get_input_channels()
    detector = ImpulseDetector(threshold=threshold)
    remaining = {c: samples_per_class for c in CLASSES}
    total_needed = samples_per_class * len(CLASSES)
    collected_total = 0

    print(f"\n{'='*65}")
    print(f" [COLLECT - RANDOMIZED GROUP MODE] (Channels: {channels})")
    print(f" Target: {samples_per_class} samples per class ({total_needed} total across {len(CLASSES)} classes)")
    print(f" Classes will cycle randomly in groups of 3 or 4 taps.")
    print(f" Press Ctrl+C at any time to exit safely.")
    print(f"{'='*65}\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=channels, callback=detector.audio_callback
        ):
            while any(v > 0 for v in remaining.values()):
                active_classes = [c for c, rem in remaining.items() if rem > 0]
                current_class = random.choice(active_classes)
                group_size = min(random.choice([3, 4]), remaining[current_class])

                instructions = {
                    "left_tap": "Tap on the LEFT palm rest",
                    "right_tap": "Tap on the RIGHT palm rest",
                    "noise": "Produce ambient noise / key clicks / desk vibrations"
                }
                desc = instructions.get(current_class, f"Perform {current_class}")

                print(f"\n{'*'*65}")
                print(f">>> TARGET: [ {current_class.upper()} ] <<<")
                print(f"Action: {desc}")
                print(f"Group quota: {group_size} tap(s) | {remaining[current_class]} remaining for this class")
                print(f"{'*'*65}")

                for i in range(group_size):
                    waveform = detector.event_queue.get()
                    save_dir = DATA_DIR / current_class
                    saved_file = save_sample(save_dir, current_class, waveform)
                    remaining[current_class] -= 1
                    collected_total += 1
                    peak = float(np.max(np.abs(waveform)))
                    print(f"  [{i+1}/{group_size}] -> {saved_file.name} (Peak: {peak:.3f}) | Progress: {collected_total}/{total_needed}")

    except KeyboardInterrupt:
        print(f"\n\n[Collection Stopped by User]")
        print("Summary of samples recorded this session:")
        for c in CLASSES:
            done = samples_per_class - remaining[c]
            print(f"  - {c:10s}: {done}/{samples_per_class} captured")
        return

    print(f"\n{'='*65}")
    print(f"Successfully collected all {total_needed} samples across {len(CLASSES)} classes!")
    print(f"{'='*65}\n")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect training taps with randomized group interleaving to eliminate session bias"
    )
    parser.add_argument(
        "--label",
        "--class",
        dest="label",
        choices=CLASSES + ["all"],
        default="all",
        help="Target class label (default: 'all' runs randomized group interleaving)"
    )
    parser.add_argument(
        "--samples",
        "--count",
        dest="samples",
        type=int,
        default=50,
        help="Number of taps to record per class (default: 50)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RMS_THRESHOLD,
        help="Mic RMS trigger threshold"
    )
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.label == "all":
        collect_group_interleaved(args.samples, args.threshold)
    else:
        collect_single_class(args.label, args.samples, args.threshold)
