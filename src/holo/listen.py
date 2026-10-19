import time
import argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
import joblib
from pynput import keyboard
from .config import SAMPLE_RATE, BLOCK_SIZE, RMS_THRESHOLD, MODEL_PATH
from .detector import ImpulseDetector
from .features import extract_features

def run_listener(model_file: Path, threshold: float, confidence_min: float, dry_run: bool):
    saved = joblib.load(model_file)
    model = saved["model"]
    model_name = saved.get("model_name", "CustomModel")
    feature_set = saved["feature_set"]
    classes = saved["classes"]

    kb_controller = keyboard.Controller()
    last_keypress_time = 0.0
    typing_suppression_window = 0.15

    def on_press(key):
        nonlocal last_keypress_time
        last_keypress_time = time.time()

    kb_listener = keyboard.Listener(on_press=on_press)
    kb_listener.start()

    def trigger_macro(action_label: str):
        if dry_run:
            print(f"  [DRY-RUN] Would have triggered shortcut for: {action_label}")
            return
        if action_label == "left_tap":
            print("  >>> ACTION: Left Palm Rest -> Copy (Cmd+C)")
            with kb_controller.pressed(keyboard.Key.cmd):
                kb_controller.tap("c")
        elif action_label == "right_tap":
            print("  >>> ACTION: Right Palm Rest -> Paste (Cmd+V)")
            with kb_controller.pressed(keyboard.Key.cmd):
                kb_controller.tap("v")

    detector = ImpulseDetector(threshold=threshold)
    mode_label = "DRY-RUN TEST MODE" if dry_run else "LIVE MACRO MODE"
    print(f"\n[LISTEN - {mode_label}] Loaded {model_name} ({feature_set} features) from {model_file.name}")
    print(f"Confidence Threshold: {confidence_min:.2f} | Tap left or right palm rest. Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, callback=detector.audio_callback
    ):
        while True:
            waveform = detector.event_queue.get()

            if (time.time() - last_keypress_time) < typing_suppression_window:
                print("  [Ignored: Active keyboard typing]")
                continue

            t0 = time.perf_counter()
            feats = extract_features(waveform, feature_set=feature_set).reshape(1, -1)
            probs = model.predict_proba(feats)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_label = classes[pred_idx]
            latency_ms = (time.perf_counter() - t0) * 1000

            prob_str = " | ".join([f"{c}: {p:.2f}" for c, p in zip(classes, probs)])
            print(f"Pred: {pred_label:10s} (Conf: {confidence:.2f}) [{prob_str}] | {latency_ms:.1f} ms")

            if pred_label != "noise" and confidence >= confidence_min:
                trigger_macro(pred_label)
                last_keypress_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test or run a trained AcousticTap model")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to a .joblib model file")
    parser.add_argument("--threshold", type=float, default=RMS_THRESHOLD, help="Mic RMS trigger threshold")
    parser.add_argument("--confidence", type=float, default=0.75, help="Minimum confidence to trigger")
    parser.add_argument("--dry-run", action="store_true", help="Test predictions without firing macOS shortcuts")
    args = parser.parse_args()
    run_listener(args.model, args.threshold, args.confidence, args.dry_run)
