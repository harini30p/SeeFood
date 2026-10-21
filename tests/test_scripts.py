"""Unit tests for SeeFood export and benchmark script utilities and CLI interfaces.

Covers:
- scripts/export_tflite.py path resolution, model configs, and argument parsing
- scripts/benchmark_models.py path resolution, argument parsing, label evaluation, and metrics
- Directory validation and error handling for missing real-dataset paths
"""

import argparse
from pathlib import Path
import pytest

from scripts.export_tflite import resolve_paths as resolve_export_paths, MODEL_CONFIGS as EXPORT_CONFIGS
from scripts.benchmark_models import (
    resolve_model_paths,
    MODEL_CONFIGS as BENCHMARK_CONFIGS,
    three_way_label,
    compute_binary_metrics,
    collect_image_files,
)


class TestExportScriptUtils:
    """Tests for export_tflite.py logic and CLI path resolution."""

    def test_export_resolve_paths_augmented(self):
        """Augmented variant should resolve to augmented keras and augmented tflite targets."""
        source, target = resolve_export_paths("augmented")
        assert source.name == "seefood_augmented_best.keras"
        assert target.name == "seefood_augmented.tflite"

    def test_export_resolve_paths_baseline(self):
        """Baseline variant should resolve to baseline keras and baseline tflite targets."""
        source, target = resolve_export_paths("baseline")
        assert source.name == "seefood_best.keras"
        assert target.name == "seefood.tflite"

    def test_export_safe_distinct_targets(self):
        """Augmented and baseline targets must never point to the same file."""
        _, aug_target = resolve_export_paths("augmented")
        _, base_target = resolve_export_paths("baseline")
        assert aug_target != base_target
        assert aug_target.name != base_target.name

    def test_export_invalid_variant_raises_error(self):
        """Invalid model variant name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model variant"):
            resolve_export_paths("invalid_variant")


class TestBenchmarkScriptUtils:
    """Tests for benchmark_models.py logic, metric calculation, and path resolution."""

    def test_benchmark_resolve_paths_augmented(self):
        """Augmented variant should pair augmented keras with augmented tflite."""
        k_path, t_path = resolve_model_paths("augmented")
        assert k_path.name == "seefood_augmented_best.keras"
        assert t_path.name == "seefood_augmented.tflite"

    def test_benchmark_resolve_paths_baseline(self):
        """Baseline variant should pair baseline keras with baseline tflite."""
        k_path, t_path = resolve_model_paths("baseline")
        assert k_path.name == "seefood_best.keras"
        assert t_path.name == "seefood.tflite"

    def test_benchmark_invalid_variant_raises_error(self):
        """Invalid model variant name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model variant"):
            resolve_model_paths("nonexistent")

    def test_three_way_label_logic(self):
        """three_way_label should correctly apply low/high boundaries."""
        assert three_way_label(0.70, 0.35, 0.65) == "Hotdog"
        assert three_way_label(0.65, 0.35, 0.65) == "Hotdog"
        assert three_way_label(0.20, 0.35, 0.65) == "Not Hotdog"
        assert three_way_label(0.35, 0.35, 0.65) == "Not Hotdog"
        assert three_way_label(0.50, 0.35, 0.65) == "Uncertain"
        assert three_way_label(0.351, 0.35, 0.65) == "Uncertain"
        assert three_way_label(0.649, 0.35, 0.65) == "Uncertain"

    def test_compute_binary_metrics_calculation(self):
        """Verify precision, recall, and f1 computation accuracy."""
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 1]

        # TP=2, FN=1 (true 1s: indices 0, 1, 4; preds: 1, 0, 1)
        # FP=1, TN=1 (true 0s: indices 2, 3; preds: 0, 1)
        res = compute_binary_metrics(y_true, y_pred)
        assert res["accuracy"] == pytest.approx(3 / 5)
        assert res["confusion_matrix"]["tp"] == 2
        assert res["confusion_matrix"]["fp"] == 1
        assert res["confusion_matrix"]["fn"] == 1
        assert res["confusion_matrix"]["tn"] == 1

        # Hotdog: Precision = 2 / 3, Recall = 2 / 3, F1 = 2/3
        assert res["hotdog"]["precision"] == pytest.approx(2 / 3)
        assert res["hotdog"]["recall"] == pytest.approx(2 / 3)
        assert res["hotdog"]["f1"] == pytest.approx(2 / 3)

    def test_collect_image_files_missing_dir_exits(self, tmp_path: Path):
        """Missing or incomplete dataset folder structure should safely exit with SystemExit."""
        fake_dir = tmp_path / "empty_dataset"
        fake_dir.mkdir()

        # Missing class subdirectories should trigger SystemExit
        with pytest.raises(SystemExit):
            collect_image_files(fake_dir)
