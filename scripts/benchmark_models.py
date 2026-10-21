"""Phase 2 benchmark and evaluation script for SeeFood.

Supports:
1. Synthetic-input conversion-consistency benchmark (latency, size, agreement)
2. Real-image validation/test dataset evaluation (--data-dir <path>)

CLI Options:
    --model augmented   -> compares models/seefood_augmented_best.keras against models/seefood_augmented.tflite
    --model baseline    -> compares models/seefood_best.keras against models/seefood.tflite
    --data-dir <path>   -> evaluates both models on real images organized in 'hot_dog' and 'not_hot_dog' folders
    --timed-runs <int>  -> number of timed inference iterations for latency (default: 50)
    --warmup-runs <int> -> number of warmup runs (default: 5)

Important:
- Synthetic benchmarking tests numerical conversion consistency only and does NOT measure real-world accuracy.
- Real-image evaluation requires a directory containing 'hot_dog' and 'not_hot_dog' subdirectories.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

# Try importing backend preprocessing so exact image transformations are reused
PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from backend.preprocessing import preprocess_image
except ImportError:
    # Fallback if running outside project root
    sys.path.insert(0, str(PROJECT_ROOT))
    from backend.preprocessing import preprocess_image

# ---------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "baseline": {
        "keras": "seefood_best.keras",
        "tflite": "seefood.tflite",
        "desc": "Baseline transfer-learning model",
    },
    "augmented": {
        "keras": "seefood_augmented_best.keras",
        "tflite": "seefood_augmented.tflite",
        "desc": "Data-augmented transfer-learning model (active in backend)",
    },
}

THRESHOLD_PATH = PROJECT_ROOT / "config" / "thresholds.json"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Label encoding convention verified across backend and training setup:
# Sigmoid output represents P(Hotdog).
# P >= 0.5 -> Hotdog (class 1); P < 0.5 -> Not Hotdog (class 0)
CLASS_LABELS = {
    "hot_dog": 1,
    "not_hot_dog": 0,
}


def resolve_model_paths(model_variant: str = "augmented") -> Tuple[Path, Path]:
    """Return (keras_path, tflite_path) for the chosen model variant."""
    if model_variant not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model variant '{model_variant}'. Choose from: {list(MODEL_CONFIGS.keys())}"
        )
    cfg = MODEL_CONFIGS[model_variant]
    keras_path = PROJECT_ROOT / "models" / cfg["keras"]
    tflite_path = PROJECT_ROOT / "models" / cfg["tflite"]
    return keras_path, tflite_path


def load_thresholds() -> Tuple[float, float]:
    """Load provisional low/high thresholds from config/thresholds.json.

    Falls back to (0.35, 0.65) if the file is missing or invalid.
    """
    defaults = (0.35, 0.65)
    if not THRESHOLD_PATH.exists():
        print(f"[WARN]  {THRESHOLD_PATH} not found. Using defaults {defaults}.")
        return defaults
    try:
        cfg = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
        lo = float(cfg["threshold_low"])
        hi = float(cfg["threshold_high"])
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError("Thresholds out of range.")
        return lo, hi
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN]  Could not parse thresholds ({exc}). Using defaults {defaults}.")
        return defaults


def three_way_label(prob: float, lo: float, hi: float) -> str:
    """Return 'Hotdog', 'Not Hotdog', or 'Uncertain' given a probability."""
    if prob >= hi:
        return "Hotdog"
    if prob <= lo:
        return "Not Hotdog"
    return "Uncertain"


def sep(char: str = "=", width: int = 65) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_keras(path: Path):
    import tensorflow as tf  # noqa: PLC0415
    if not path.exists():
        print(f"[ERROR] Keras model not found at: {path}")
        print(f"        Place '{path.name}' in the 'models/' directory and retry.")
        sys.exit(1)
    print(f"[INFO]  Loading Keras model  : {path.name}")
    model = tf.keras.models.load_model(path)
    return model


def load_tflite(path: Path, variant: str):
    import tensorflow as tf  # noqa: PLC0415
    if not path.exists():
        print(f"[ERROR] TFLite model not found at: {path}")
        print(f"        Run 'python scripts/export_tflite.py --model {variant}' first.")
        sys.exit(1)
    print(f"[INFO]  Loading TFLite model : {path.name}")
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def keras_predict(model, batch: np.ndarray) -> float:
    """Run a single (1, 224, 224, 3) batch through the Keras model."""
    out = model(batch, training=False)
    return float(out.numpy().ravel()[0])


def tflite_predict(interpreter, batch: np.ndarray) -> float:
    """Run a single (1, 224, 224, 3) batch through the TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    interpreter.resize_tensor_input(input_index, batch.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, batch.astype(np.float32))
    interpreter.invoke()
    return float(interpreter.get_tensor(output_index).ravel()[0])


# ---------------------------------------------------------------------------
# Synthetic Benchmark Mode
# ---------------------------------------------------------------------------

def make_synthetic_inputs(n: int, seed: int = 42) -> np.ndarray:
    """Return n deterministic float32 inputs shaped (n, 1, 224, 224, 3) in [-1, 1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 1, 224, 224, 3)).astype(np.float32)


def benchmark_latency(
    label: str,
    predict_fn,
    inputs: np.ndarray,
    n_warmup: int,
    n_timed: int,
) -> dict:
    """Run warm-up then timed inferences and return latency statistics (ms)."""
    print(f"  [{label}] Warming up ({n_warmup} runs) ...", end="", flush=True)
    for i in range(n_warmup):
        predict_fn(inputs[i % len(inputs)])
    print(" done.")

    print(f"  [{label}] Timing {n_timed} runs ...", end="", flush=True)
    times_ms: List[float] = []
    for i in range(n_timed):
        t0 = time.perf_counter()
        predict_fn(inputs[i % len(inputs)])
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    print(" done.")

    return {
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
    }


def benchmark_agreement(
    keras_model,
    tflite_interp,
    inputs: np.ndarray,
    lo: float,
    hi: float,
) -> dict:
    """Compare sigmoid probabilities and labels from both models on synthetic inputs."""
    keras_probs: List[float] = []
    tflite_probs: List[float] = []
    binary_agreements = 0
    label_agreements = 0
    disagreements: List[dict] = []

    for idx, batch in enumerate(inputs):
        kp = keras_predict(keras_model, batch)
        tp = tflite_predict(tflite_interp, batch)
        keras_probs.append(kp)
        tflite_probs.append(tp)

        if (kp >= 0.5) == (tp >= 0.5):
            binary_agreements += 1

        k_label = three_way_label(kp, lo, hi)
        t_label = three_way_label(tp, lo, hi)
        if k_label == t_label:
            label_agreements += 1
        else:
            disagreements.append(
                {
                    "sample": idx,
                    "keras_prob": kp,
                    "tflite_prob": tp,
                    "keras_label": k_label,
                    "tflite_label": t_label,
                }
            )

    n = len(inputs)
    abs_diffs = np.abs(np.array(keras_probs) - np.array(tflite_probs))

    return {
        "n": n,
        "mean_abs_diff": float(abs_diffs.mean()),
        "max_abs_diff": float(abs_diffs.max()),
        "binary_match_pct": binary_agreements / n * 100.0,
        "label_match_pct": label_agreements / n * 100.0,
        "disagreements": disagreements,
    }


def run_synthetic_benchmark(
    keras_model,
    tflite_interp,
    keras_path: Path,
    tflite_path: Path,
    lo: float,
    hi: float,
    n_warmup: int,
    n_timed: int,
) -> None:
    """Execute synthetic consistency and latency evaluation."""
    n_inputs = max(n_warmup + n_timed, 20)
    inputs = make_synthetic_inputs(n_inputs)

    # Size comparison
    src = keras_path.stat().st_size
    tgt = tflite_path.stat().st_size
    pct = (1.0 - tgt / src) * 100.0
    print()
    print(sep())
    print("  MODEL FILE SIZES")
    print(sep())
    print(f"  {'Model':<30} {'Bytes':>12}  {'MB':>7}")
    print(f"  {'-'*30} {'-'*12}  {'-'*7}")
    print(f"  {keras_path.name:<30} {src:>12,}  {src/(1024**2):>7.2f}")
    print(f"  {tflite_path.name:<30} {tgt:>12,}  {tgt/(1024**2):>7.2f}")
    print(f"  {'Size reduction':<30} {'':>12}  {pct:>6.1f}%")
    print(sep())

    # Latency
    print("\n[INFO]  Running Keras latency benchmark (synthetic inputs) ...")
    keras_lat = benchmark_latency("Keras", lambda x: keras_predict(keras_model, x), inputs, n_warmup, n_timed)

    print("\n[INFO]  Running TFLite latency benchmark (synthetic inputs) ...")
    tflite_lat = benchmark_latency("TFLite", lambda x: tflite_predict(tflite_interp, x), inputs, n_warmup, n_timed)

    speedup = keras_lat["mean_ms"] / tflite_lat["mean_ms"] if tflite_lat["mean_ms"] > 0 else float("nan")
    print()
    print(sep())
    print(f"  INFERENCE LATENCY (warm-up={n_warmup}, timed={n_timed} runs)")
    print(sep())
    print(f"  {'Metric':<20} {'Keras (ms)':>15} {'TFLite (ms)':>15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    for key, label in [("mean_ms", "Mean"), ("median_ms", "Median"), ("p95_ms", "95th pct")]:
        print(f"  {label:<20} {keras_lat[key]:>15.2f} {tflite_lat[key]:>15.2f}")
    print(f"  {'TFLite speedup':<20} {'':>15} {speedup:>14.2f}x")
    print(sep())

    # Agreement
    print(f"\n[INFO]  Comparing predictions on 20 synthetic inputs ...")
    ag = benchmark_agreement(keras_model, tflite_interp, inputs[:20], lo, hi)
    print()
    print(sep())
    print(f"  SYNTHETIC PREDICTION AGREEMENT (n={ag['n']}, seed=42)")
    print(sep())
    print(f"  Mean absolute prob. difference   : {ag['mean_abs_diff']:.6f}")
    print(f"  Max  absolute prob. difference   : {ag['max_abs_diff']:.6f}")
    print(f"  Binary decisions match (>=0.5)   : {ag['binary_match_pct']:.1f}%")
    print(f"  3-way label match (lo={lo}, hi={hi}) : {ag['label_match_pct']:.1f}%")
    if ag["disagreements"]:
        print()
        print(f"  Three-way label disagreements ({len(ag['disagreements'])}):")
        for d in ag["disagreements"]:
            print(
                f"    sample {d['sample']:>2}: keras={d['keras_prob']:.4f} ({d['keras_label']:<10})"
                f"  tflite={d['tflite_prob']:.4f} ({d['tflite_label']})"
            )
    else:
        print("  Three-way label disagreements    : none")
    print(sep())

    print()
    print(sep("!", 65))
    print("  NOTICE: SYNTHETIC INPUT BENCHMARK COMPLETED")
    print(sep("!", 65))
    print("  Inputs were random float32 arrays in [-1, 1].")
    print("  This benchmark verifies numerical conversion consistency only.")
    print("  It does NOT measure classification accuracy on real food photos.")
    print("  To evaluate on real images, supply: --data-dir <path/to/dataset>")
    print(sep("!", 65))


# ---------------------------------------------------------------------------
# Real Validation/Test Evaluation Mode
# ---------------------------------------------------------------------------

def collect_image_files(data_dir: Path) -> List[Tuple[Path, int]]:
    """Scan data_dir for 'hot_dog' and 'not_hot_dog' subdirectories.

    Returns:
        List[Tuple[Path, int]]: (image_path, ground_truth_label)
    """
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"[ERROR] Specified data directory does not exist or is not a directory: {data_dir}")
        sys.exit(1)

    items: List[Tuple[Path, int]] = []
    missing_classes = []

    for class_name, label in CLASS_LABELS.items():
        class_folder = data_dir / class_name
        if not class_folder.exists() or not class_folder.is_dir():
            missing_classes.append(class_name)
            continue

        for file_path in class_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                items.append((file_path, label))

    if missing_classes:
        print(f"[ERROR] Missing required class subdirectory in {data_dir}: {missing_classes}")
        print("        Expected layout:")
        print(f"          {data_dir}/hot_dog/")
        print(f"          {data_dir}/not_hot_dog/")
        sys.exit(1)

    if not items:
        print(f"[ERROR] No supported images found in {data_dir} (extensions: {SUPPORTED_EXTENSIONS})")
        sys.exit(1)

    return items


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    """Compute precision, recall, f1 for binary classes."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Hotdog metrics (class 1)
    p1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0.0

    # Not Hotdog metrics (class 0)
    p0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    r0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "hotdog": {"precision": p1, "recall": r1, "f1": f1_1, "support": tp + fn},
        "not_hotdog": {"precision": p0, "recall": r0, "f1": f1_0, "support": tn + fp},
    }


def run_real_image_evaluation(
    keras_model,
    tflite_interp,
    keras_path: Path,
    tflite_path: Path,
    data_dir: Path,
    lo: float,
    hi: float,
) -> None:
    """Evaluate both Keras and TFLite models on identical real dataset images."""
    items = collect_image_files(data_dir)
    total_images = len(items)

    print()
    print(sep())
    print(f"  REAL-IMAGE EVALUATION ON: {data_dir.resolve()}")
    print(f"  Total valid images found: {total_images}")
    print(sep())

    y_true: List[int] = []
    k_probs: List[float] = []
    t_probs: List[float] = []

    k_times: List[float] = []
    t_times: List[float] = []
    skipped_images = 0

    for idx, (img_path, ground_truth) in enumerate(items, start=1):
        try:
            with Image.open(img_path) as pil_img:
                # Reuse the exact preprocessing from backend
                batch = preprocess_image(pil_img)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            print(f"[WARN] Skipping corrupted/unreadable image '{img_path.name}': {exc}")
            skipped_images += 1
            continue

        # Timed Keras inference
        t0 = time.perf_counter()
        kp = keras_predict(keras_model, batch)
        k_times.append((time.perf_counter() - t0) * 1000.0)

        # Timed TFLite inference
        t0 = time.perf_counter()
        tp = tflite_predict(tflite_interp, batch)
        t_times.append((time.perf_counter() - t0) * 1000.0)

        y_true.append(ground_truth)
        k_probs.append(kp)
        t_probs.append(tp)

        if idx % 100 == 0 or idx == total_images:
            print(f"  Processed {idx}/{total_images} images...", flush=True)

    evaluated_count = len(y_true)
    if evaluated_count == 0:
        print("[ERROR] No images could be evaluated successfully.")
        sys.exit(1)

    # Decisions
    k_bin_pred = [1 if p >= 0.5 else 0 for p in k_probs]
    t_bin_pred = [1 if p >= 0.5 else 0 for p in t_probs]

    k_metrics = compute_binary_metrics(y_true, k_bin_pred)
    t_metrics = compute_binary_metrics(y_true, t_bin_pred)

    binary_matches = sum(1 for kp, tp in zip(k_bin_pred, t_bin_pred) if kp == tp)
    three_way_matches = sum(
        1 for kp, tp in zip(k_probs, t_probs)
        if three_way_label(kp, lo, hi) == three_way_label(tp, lo, hi)
    )

    diffs = np.abs(np.array(k_probs) - np.array(t_probs))

    # Print results
    print()
    print(sep())
    print(f"  EVALUATION ACCURACY & REPORT (N={evaluated_count} images)")
    print(sep())
    print(f"  {'Metric':<25} {'Keras (' + keras_path.name + ')':<30} {'TFLite (' + tflite_path.name + ')'}")
    print(f"  {'-'*25} {'-'*30} {'-'*30}")
    print(f"  {'Overall Accuracy':<25} {k_metrics['accuracy']*100:>28.2f}% {t_metrics['accuracy']*100:>28.2f}%")
    print()
    print("  Hotdog Class (1):")
    print(f"    Precision             : {k_metrics['hotdog']['precision']*100:>28.2f}% {t_metrics['hotdog']['precision']*100:>28.2f}%")
    print(f"    Recall                : {k_metrics['hotdog']['recall']*100:>28.2f}% {t_metrics['hotdog']['recall']*100:>28.2f}%")
    print(f"    F1-Score              : {k_metrics['hotdog']['f1']*100:>28.2f}% {t_metrics['hotdog']['f1']*100:>28.2f}%")
    print()
    print("  Not Hotdog Class (0):")
    print(f"    Precision             : {k_metrics['not_hotdog']['precision']*100:>28.2f}% {t_metrics['not_hotdog']['precision']*100:>28.2f}%")
    print(f"    Recall                : {k_metrics['not_hotdog']['recall']*100:>28.2f}% {t_metrics['not_hotdog']['recall']*100:>28.2f}%")
    print(f"    F1-Score              : {k_metrics['not_hotdog']['f1']*100:>28.2f}% {t_metrics['not_hotdog']['f1']*100:>28.2f}%")
    print()
    print("  Confusion Matrix [TP, FP, FN, TN]:")
    k_cm = k_metrics['confusion_matrix']
    t_cm = t_metrics['confusion_matrix']
    print(f"    Keras                 : TP={k_cm['tp']}, FP={k_cm['fp']}, FN={k_cm['fn']}, TN={k_cm['tn']}")
    print(f"    TFLite                : TP={t_cm['tp']}, FP={t_cm['fp']}, FN={t_cm['fn']}, TN={t_cm['tn']}")
    print(sep())

    print()
    print(sep())
    print("  MODEL-TO-MODEL CONSISTENCY ON REAL IMAGES")
    print(sep())
    print(f"  Binary Agreement (p >= 0.5)      : {binary_matches / evaluated_count * 100.0:.2f}% ({binary_matches}/{evaluated_count})")
    print(f"  3-Way Agreement (lo={lo}, hi={hi}): {three_way_matches / evaluated_count * 100.0:.2f}% ({three_way_matches}/{evaluated_count})")
    print(f"  Mean Absolute Prob. Difference   : {float(diffs.mean()):.6f}")
    print(f"  Max  Absolute Prob. Difference   : {float(diffs.max()):.6f}")
    print(sep())

    print()
    print(sep())
    print("  MEASURED INFERENCE LATENCY ON REAL IMAGES")
    print(sep())
    print(f"  {'Metric':<20} {'Keras (ms)':>15} {'TFLite (ms)':>15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    print(f"  {'Mean':<20} {np.mean(k_times):>15.2f} {np.mean(t_times):>15.2f}")
    print(f"  {'Median':<20} {np.median(k_times):>15.2f} {np.median(t_times):>15.2f}")
    print(f"  {'95th Percentile':<20} {np.percentile(k_times, 95):>15.2f} {np.percentile(t_times, 95):>15.2f}")
    speedup = np.mean(k_times) / np.mean(t_times) if np.mean(t_times) > 0 else float("nan")
    print(f"  {'TFLite speedup':<20} {'':>15} {speedup:>14.2f}x")
    print(sep())


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and evaluate SeeFood Keras and TFLite model pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["augmented", "baseline"],
        default="augmented",
        help="Select model pair to evaluate ('augmented' or 'baseline').",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional path to real dataset directory containing 'hot_dog' and 'not_hot_dog' subfolders.",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=50,
        help="Number of timed inference iterations for synthetic latency test.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup iterations for synthetic latency test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"\n[INFO]  Initializing benchmark for '{args.model}' model pair...")

    keras_path, tflite_path = resolve_model_paths(args.model)
    lo, hi = load_thresholds()

    keras_model = load_keras(keras_path)
    tflite_interp = load_tflite(tflite_path, args.model)

    if args.data_dir:
        # Real dataset evaluation mode
        run_real_image_evaluation(
            keras_model=keras_model,
            tflite_interp=tflite_interp,
            keras_path=keras_path,
            tflite_path=tflite_path,
            data_dir=args.data_dir,
            lo=lo,
            hi=hi,
        )
    else:
        # Synthetic consistency benchmark mode
        run_synthetic_benchmark(
            keras_model=keras_model,
            tflite_interp=tflite_interp,
            keras_path=keras_path,
            tflite_path=tflite_path,
            lo=lo,
            hi=hi,
            n_warmup=args.warmup_runs,
            n_timed=args.timed_runs,
        )


if __name__ == "__main__":
    main()
