import time
import joblib
import numpy as np
import sounddevice as sd
from .config import BLOCK_SIZE, CALIBRATION_BLOCKS, CONFIDENCE_THRESHOLD, ENABLE_SHORTCUTS, MODEL_PATH, SAMPLE_RATE, ULTRASONIC_2FA_ENABLED, ULTRASONIC_ENABLED
from .detector import ImpulseDetector
from .features import extract_features
from .ultrasonic import UltrasonicSonarGate

def run_listener():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train first.")
    bundle = joblib.load(MODEL_PATH)
    model, triangulator, fourier_weight, classes = bundle["model"], bundle["triangulator"], float(bundle.get("fourier_weight", 0.50)), list(bundle["classes"])
    detector, sonar = ImpulseDetector(sr=SAMPLE_RATE), UltrasonicSonarGate(sr=SAMPLE_RATE)
    if ULTRASONIC_ENABLED:
        sonar.start_emitter()
        print("[Sonar] Emitter active. Keep hands off palm rests for calibration...")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE, callback=detector.audio_callback):
            if ULTRASONIC_ENABLED:
                while len(detector.calibration_buffer) < CALIBRATION_BLOCKS: time.sleep(0.05)
                sonar.calibrate_baseline(list(detector.calibration_buffer))
                print("[Sonar] Calibration locked! Ready for taps.")
            while True:
                try:
                    raw_waveform = detector.event_queue.get(timeout=0.1)
                except Exception:
                    continue
                p_v2 = (1.0 - fourier_weight) * model.predict_proba(extract_features(raw_waveform, sr=SAMPLE_RATE).reshape(1, -1))[0] + fourier_weight * triangulator.predict_proba(raw_waveform)[0]
                probs = {cls: float(p_v2[i]) for i, cls in enumerate(classes)}
                us = sonar.evaluate_window(raw_waveform)
                if ULTRASONIC_ENABLED and ULTRASONIC_2FA_ENABLED and us["active"]:
                    for k in probs:
                        if "left" in k: probs[k] *= (0.5 + 0.5 * us["gate_left"])
                        elif "right" in k: probs[k] *= (0.5 + 0.5 * us["gate_right"])
                pred_cls = max(probs, key=probs.get)
                if "noise" not in pred_cls and probs[pred_cls] >= CONFIDENCE_THRESHOLD:
                    print(f">>> TRIGGER: {pred_cls.upper()} | Conf: {probs[pred_cls]:.2f} | Gates L/R: {us['gate_left']:.2f}/{us['gate_right']:.2f}")
                else:
                    print(f"    [Rejected] Pred: {pred_cls} ({probs[pred_cls]:.2f}) | Gates L/R: {us['gate_left']:.2f}/{us['gate_right']:.2f}")
    except KeyboardInterrupt:
        print("Stopping listener...")
    finally:
        sonar.stop_emitter()

if __name__ == "__main__":
    run_listener()
