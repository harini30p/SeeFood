"""Phase 1 TFLite export script for SeeFood.

Loads the existing Keras model, exports an optimized TensorFlow Lite model,
verifies the output, and prints tensor shapes and sizes.

Keras 3 + TF 2.16.x note
--------------------------
TFLiteConverter.from_keras_model() is a Keras 2 / tf.keras API. Under
Keras 3, its internal tracing path produces ReadVariableOp nodes without the
'value' attribute that TF-Lite's MLIR frontend requires, causing:

    LLVM ERROR: Failed to infer result type(s).

The fix is a two-step process:
  1. Export a proper TF2 SavedModel via Keras 3's model.export().
  2. Convert with TFLiteConverter.from_saved_model().

The temporary SavedModel directory is created inside tempfile.TemporaryDirectory
and is cleaned up automatically after conversion.

Usage (from the project root):
    .venv\\Scripts\\python.exe scripts/export_tflite.py

Nothing in backend/, frontend/, thresholds, tests, or the original .keras
model is touched.
"""

import sys
import tempfile
from pathlib import Path


def resolve_paths() -> tuple[Path, Path]:
    """Return (source .keras path, target .tflite path) relative to project root."""
    project_root = Path(__file__).resolve().parent.parent
    source_path = project_root / "models" / "seefood_best.keras"
    target_path = project_root / "models" / "seefood.tflite"
    return source_path, target_path


def load_keras_model(source_path: Path):
    """Load the Keras model from disk.

    Raises:
        SystemExit: If the model file does not exist.
    """
    if not source_path.exists():
        print(f"[ERROR] Source model not found: {source_path}")
        print("        Place 'seefood_best.keras' inside the 'models/' directory and retry.")
        sys.exit(1)

    print(f"[INFO]  Loading Keras model from: {source_path}")

    import tensorflow as tf  # noqa: PLC0415

    model = tf.keras.models.load_model(source_path)
    print(f"[INFO]  Model loaded successfully.  Input shape: {model.input_shape}")
    return model


def export_tflite(model, target_path: Path) -> bytes:
    """Convert the Keras 3 model to TFLite via a temporary TF2 SavedModel.

    Keras 3's model.export() writes a proper TF2 SavedModel that
    TFLiteConverter.from_saved_model() can process without the MLIR
    ReadVariableOp error that from_keras_model() triggers.

    tf.lite.Optimize.DEFAULT applies dynamic-range quantization: weights are
    stored as INT8 and dequantised to Float32 at runtime, reducing file size
    by roughly 70-75% with negligible accuracy loss.

    Args:
        model: Loaded tf.keras.Model (Keras 3).
        target_path: Destination path for the .tflite flatbuffer.

    Returns:
        bytes: Raw TFLite flatbuffer content written to disk.

    Raises:
        SystemExit: On export or conversion failure.
    """
    import tensorflow as tf  # noqa: PLC0415

    # ------------------------------------------------------------------ #
    # Step 1 - export a proper TF2 SavedModel via the Keras 3 API         #
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory(prefix="seefood_savedmodel_") as tmp_dir:
        print(f"[INFO]  Exporting Keras 3 model to temporary SavedModel: {tmp_dir}")
        try:
            model.export(tmp_dir)
        except Exception as exc:
            print(f"[ERROR] model.export() failed: {exc}")
            sys.exit(1)

        print("[INFO]  SavedModel export complete.")

        # ------------------------------------------------------------------ #
        # Step 2 - convert from SavedModel (bypasses Keras 2 tracing path)   #
        # ------------------------------------------------------------------ #
        print("[INFO]  Configuring TFLiteConverter from SavedModel ...")
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
        except Exception as exc:
            print(f"[ERROR] TFLiteConverter.from_saved_model() failed: {exc}")
            sys.exit(1)

        # Dynamic-range quantization: weights -> INT8, activations -> Float32
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        print("[INFO]  Running TFLite conversion (this may take a few seconds) ...")
        try:
            tflite_flatbuffer = converter.convert()
        except Exception as exc:
            print(f"[ERROR] converter.convert() failed: {exc}")
            sys.exit(1)

        # TemporaryDirectory cleaned up automatically on context exit
        print("[INFO]  Temporary SavedModel directory cleaned up.")

    # ------------------------------------------------------------------ #
    # Step 3 - write the flatbuffer to models/seefood.tflite              #
    # ------------------------------------------------------------------ #
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_path.write_bytes(tflite_flatbuffer)
    except OSError as exc:
        print(f"[ERROR] Could not write TFLite model to {target_path}: {exc}")
        sys.exit(1)

    print(f"[INFO]  TFLite model written to: {target_path}")
    return tflite_flatbuffer


def report_sizes(source_path: Path, target_path: Path) -> None:
    """Print file sizes and percentage reduction."""
    src_bytes = source_path.stat().st_size
    tgt_bytes = target_path.stat().st_size
    reduction_pct = (1.0 - tgt_bytes / src_bytes) * 100.0

    print()
    print("=" * 57)
    print("  File size comparison")
    print("=" * 57)
    print(f"  Source  (.keras):   {src_bytes:>12,} bytes  ({src_bytes / (1024**2):.2f} MB)")
    print(f"  Exported (.tflite): {tgt_bytes:>12,} bytes  ({tgt_bytes / (1024**2):.2f} MB)")
    print(f"  Size reduction:     {reduction_pct:.1f}%")
    print("=" * 57)


def verify_and_inspect(target_path: Path) -> None:
    """Load the exported file with tf.lite.Interpreter and print tensor details.

    Raises:
        SystemExit: If the exported file is missing or zero bytes.
    """
    import tensorflow as tf  # noqa: PLC0415

    if not target_path.exists():
        print(f"[ERROR] Exported file not found at: {target_path}")
        sys.exit(1)

    if target_path.stat().st_size == 0:
        print(f"[ERROR] Exported file is empty: {target_path}")
        sys.exit(1)

    print("\n[INFO]  Verifying exported model with tf.lite.Interpreter ...")
    try:
        interpreter = tf.lite.Interpreter(model_path=str(target_path))
        interpreter.allocate_tensors()
    except Exception as exc:
        print(f"[ERROR] Interpreter verification failed: {exc}")
        sys.exit(1)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print()
    print("=" * 57)
    print("  TFLite tensor inspection")
    print("=" * 57)
    for idx, inp in enumerate(input_details):
        print(f"  Input  [{idx}]  name  : {inp['name']}")
        print(f"           shape : {inp['shape'].tolist()}")
        print(f"           dtype : {inp['dtype'].__name__}")
    print()
    for idx, out in enumerate(output_details):
        print(f"  Output [{idx}]  name  : {out['name']}")
        print(f"           shape : {out['shape'].tolist()}")
        print(f"           dtype : {out['dtype'].__name__}")
    print("=" * 57)
    print("\n[OK]    Verification passed. The TFLite model is ready for use.")


def check_gitignore(target_path: Path) -> None:
    """Confirm the .tflite file is covered by .gitignore."""
    gitignore = target_path.parent.parent / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if "*.tflite" in content or "models/*.tflite" in content:
            print("[INFO]  .gitignore already excludes models/*.tflite - file will NOT be committed.")
        else:
            print("[WARN]  models/*.tflite is NOT in .gitignore. Add it before committing.")
    else:
        print("[WARN]  No .gitignore found at project root.")


def main() -> None:
    source_path, target_path = resolve_paths()

    model = load_keras_model(source_path)
    export_tflite(model, target_path)
    report_sizes(source_path, target_path)
    verify_and_inspect(target_path)
    check_gitignore(target_path)


if __name__ == "__main__":
    main()
