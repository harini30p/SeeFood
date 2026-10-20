"""Phase 2 benchmark script for SeeFood.

Compares models/seefood_best.keras and models/seefood.tflite on:
  - Inference latency  (warm-up then timed runs)
  - File size
  - Sigmoid probability agreement on deterministic synthetic inputs
  - Three-way label agreement using provisional thresholds from config/thresholds.json

WARNING: All inputs are synthetic float32 arrays in [-1, 1].
They test conversion consistency only and do NOT establish real-world accuracy
or validate the provisional thresholds.

Usage (from the project root):
    .venv\\Scripts\\python.exe scripts/benchmark_models.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERAS_PATH = PROJECT_ROOT / "models" / "seefood_best.keras"
TFLITE_PATH = PROJECT_ROOT / "models" / "seefood.tflite"
THRESHOLD_PATH = PROJECT_ROOT / "config" / "thresholds.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WARMUP_RUNS = 5
TIMED_RUNS = 50
AGREEMENT_SAMPLES = 20
BINARY_THRESHOLD = 0.5
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_thresholds() -> tuple[float, float]:
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


def make_inputs(n: int, seed: int = RNG_SEED) -> np.ndarray:
    """Return n deterministic float32 inputs shaped (n, 1, 224, 224, 3) in [-1, 1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 1, 224, 224, 3)).astype(np.float32)


def percentile95(values: list[float]) -> float:
    return float(np.percentile(values, 95))


def sep(char: str = "=", width: int = 60) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_keras(path: Path):
    import tensorflow as tf  # noqa: PLC0415
    if not path.exists():
        print(f"[ERROR] Keras model not found: {path}")
        sys.exit(1)
    print(f"[INFO]  Loading Keras model  : {path.name}")
    model = tf.keras.models.load_model(path)
    return model


def load_tflite(path: Path):
    import tensorflow as tf  # noqa: PLC0415
    if not path.exists():
        print(f"[ERROR] TFLite model not found: {path}")
        print("        Run scripts/export_tflite.py first.")
        sys.exit(1)
    print(f"[INFO]  Loading TFLite model : {path.name}")
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def keras_predict(model, batch: np.ndarray) -> float:
    """Run a single (1, 224, 224, 3) batch through the Keras model.

    Returns the scalar sigmoid probability.
    """
    out = model(batch, training=False)
    return float(out.numpy().ravel()[0])


def tflite_predict(interpreter, batch: np.ndarray) -> float:
    """Run a single (1, 224, 224, 3) batch through the TFLite interpreter.

    Handles tensor resizing, sets the float32 input, invokes, and returns
    the scalar sigmoid probability.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    # Resize in case the interpreter was allocated with a different batch size
    interpreter.resize_tensor_input(input_index, batch.shape)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_index, batch.astype(np.float32))
    interpreter.invoke()

    return float(interpreter.get_tensor(output_index).ravel()[0])


# ---------------------------------------------------------------------------
# Benchmark: latency
# ---------------------------------------------------------------------------

def benchmark_latency(
    label: str,
    predict_fn,
    inputs: np.ndarray,
    n_warmup: int = WARMUP_RUNS,
    n_timed: int = TIMED_RUNS,
) -> dict:
    """Run warm-up then timed inferences and return latency statistics (ms)."""
    print(f"  [{label}] Warming up ({n_warmup} runs) …", end="", flush=True)
    for i in range(n_warmup):
        predict_fn(inputs[i % len(inputs)])
    print(" done.")

    print(f"  [{label}] Timing {n_timed} runs …", end="", flush=True)
    times_ms: list[float] = []
    for i in range(n_timed):
        t0 = time.perf_counter()
        predict_fn(inputs[i % len(inputs)])
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    print(" done.")

    return {
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": percentile95(times_ms),
    }


# ---------------------------------------------------------------------------
# Benchmark: probability / label agreement
# ---------------------------------------------------------------------------

def benchmark_agreement(
    keras_model,
    tflite_interp,
    inputs: np.ndarray,
    lo: float,
    hi: float,
) -> dict:
    """Compare sigmoid probabilities and labels from both models."""
    keras_probs: list[float] = []
    tflite_probs: list[float] = []
    binary_agreements = 0
    label_agreements = 0
    disagreements: list[dict] = []

    for idx, batch in enumerate(inputs):
        kp = keras_predict(keras_model, batch)
        tp = tflite_predict(tflite_interp, batch)
        keras_probs.append(kp)
        tflite_probs.append(tp)

        k_bin = kp >= BINARY_THRESHOLD
        t_bin = tp >= BINARY_THRESHOLD
        if k_bin == t_bin:
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_size_table(keras_path: Path, tflite_path: Path) -> None:
    src = keras_path.stat().st_size
    tgt = tflite_path.stat().st_size
    pct = (1.0 - tgt / src) * 100.0
    print()
    print(sep())
    print("  MODEL FILE SIZES")
    print(sep())
    print(f"  {'Model':<22} {'Bytes':>12}  {'MB':>7}")
    print(f"  {'-'*22} {'-'*12}  {'-'*7}")
    print(f"  {'seefood_best.keras':<22} {src:>12,}  {src/(1024**2):>7.2f}")
    print(f"  {'seefood.tflite':<22} {tgt:>12,}  {tgt/(1024**2):>7.2f}")
    print(f"  {'Size reduction':<22} {'':>12}  {pct:>6.1f}%")
    print(sep())


def print_latency_table(keras_stats: dict, tflite_stats: dict) -> None:
    speedup = keras_stats["mean_ms"] / tflite_stats["mean_ms"] if tflite_stats["mean_ms"] else float("nan")
    print()
    print(sep())
    print(f"  INFERENCE LATENCY  (warm-up={WARMUP_RUNS}, timed={TIMED_RUNS} runs)")
    print(sep())
    print(f"  {'Metric':<18} {'Keras (ms)':>12} {'TFLite (ms)':>12}")
    print(f"  {'-'*18} {'-'*12} {'-'*12}")
    for key, label in [("mean_ms", "Mean"), ("median_ms", "Median"), ("p95_ms", "95th pct")]:
        print(f"  {label:<18} {keras_stats[key]:>12.2f} {tflite_stats[key]:>12.2f}")
    print(f"  {'TFLite speedup':<18} {'':>12} {speedup:>11.2f}x")
    print(sep())


def print_agreement_table(ag: dict, lo: float, hi: float) -> None:
    print()
    print(sep())
    print(f"  PREDICTION AGREEMENT  (n={ag['n']} synthetic samples, seed={RNG_SEED})")
    print(sep())
    print(f"  Mean absolute prob. difference  : {ag['mean_abs_diff']:.6f}")
    print(f"  Max  absolute prob. difference  : {ag['max_abs_diff']:.6f}")
    print(f"  Binary decisions match (>=0.5)  : {ag['binary_match_pct']:.1f}%")
    print(f"  3-way label match (lo={lo}, hi={hi}): {ag['label_match_pct']:.1f}%")
    if ag["disagreements"]:
        print()
        print(f"  Three-way label disagreements ({len(ag['disagreements'])}):")
        for d in ag["disagreements"]:
            print(
                f"    sample {d['sample']:>2}: keras={d['keras_prob']:.4f} ({d['keras_label']:<10})"
                f"  tflite={d['tflite_prob']:.4f} ({d['tflite_label']})"
            )
    else:
        print("  Three-way label disagreements   : none")
    print(sep())


def print_disclaimer() -> None:
    print()
    print(sep("!", 60))
    print("  WARNING: SYNTHETIC INPUTS ONLY")
    print(sep("!", 60))
    print("  All benchmark inputs are random float32 arrays in [-1, 1].")
    print("  They verify conversion consistency between the two models.")
    print("  They do NOT measure real-world accuracy on food images,")
    print("  and they do NOT validate the provisional thresholds.")
    print("  Always evaluate on a real labelled validation set before")
    print("  deploying any threshold changes.")
    print(sep("!", 60))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(sep())
    print("  SeeFood — Phase 2 Model Benchmark")
    print(sep())

    # ---- Load thresholds ------------------------------------------------
    lo, hi = load_thresholds()
    print(f"[INFO]  Provisional thresholds  : low={lo}, high={hi}")

    # ---- Load models ----------------------------------------------------
    keras_model = load_keras(KERAS_PATH)
    tflite_interp = load_tflite(TFLITE_PATH)

    # ---- Generate synthetic inputs --------------------------------------
    # n_inputs must be >= max(WARMUP_RUNS + TIMED_RUNS, AGREEMENT_SAMPLES)
    n_inputs = max(WARMUP_RUNS + TIMED_RUNS, AGREEMENT_SAMPLES)
    inputs = make_inputs(n_inputs)
    print(f"[INFO]  Generated {n_inputs} deterministic synthetic inputs (seed={RNG_SEED}).")

    # ---- File sizes -----------------------------------------------------
    print_size_table(KERAS_PATH, TFLITE_PATH)

    # ---- Latency --------------------------------------------------------
    print("\n[INFO]  Running Keras latency benchmark …")
    keras_lat = benchmark_latency(
        "Keras",
        lambda x: keras_predict(keras_model, x),
        inputs,
    )

    print("\n[INFO]  Running TFLite latency benchmark …")
    tflite_lat = benchmark_latency(
        "TFLite",
        lambda x: tflite_predict(tflite_interp, x),
        inputs,
    )

    print_latency_table(keras_lat, tflite_lat)

    # ---- Agreement on AGREEMENT_SAMPLES inputs -------------------------
    print(f"\n[INFO]  Comparing predictions on {AGREEMENT_SAMPLES} synthetic inputs …")
    ag = benchmark_agreement(
        keras_model,
        tflite_interp,
        inputs[:AGREEMENT_SAMPLES],
        lo,
        hi,
    )
    print_agreement_table(ag, lo, hi)

    # ---- Disclaimer -----------------------------------------------------
    print_disclaimer()


if __name__ == "__main__":
    main()
