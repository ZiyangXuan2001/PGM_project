"""Shape and local experiment-manager tests for controlled DiffTraj-PGM."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import EmbeddingDifferencePGMModel  # noqa: E402
from utils import load_config  # noqa: E402
from utils.experiment_manager import (  # noqa: E402
    append_registry,
    append_train_log,
    create_run_dir,
    init_train_log,
    save_metrics,
    save_resolved_config,
    write_experiment_card,
    write_model_summary,
)


VARIANTS = [
    ("E0", "mean_pool_baseline"),
    ("E1", "diff_only"),
    ("E2", "diff_pgm"),
    ("E3", "diff_pgm_info"),
    ("E4", "diff_pgm_info_attention"),
]


class ControlledModelShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        self.B = 2
        self.T = 16
        self.d_x = 512
        self.X = torch.randn(self.B, self.T, self.d_x)

    def make_config(self, ablation_id: str, variant: str) -> dict:
        config = deepcopy(self.config)
        config["model"]["ablation_id"] = ablation_id
        config["model"]["variant"] = variant
        if ablation_id in {"E0", "E1"}:
            config["pgm_smoother"]["type"] = "none"
        else:
            config["pgm_smoother"]["type"] = "gaussian_chain"
        config["classifier"]["type"] = "attention_pool" if ablation_id == "E4" else "mlp"
        return config

    def test_e0_to_e4_logits_shape(self) -> None:
        for ablation_id, variant in VARIANTS:
            with self.subTest(ablation_id=ablation_id, variant=variant):
                config = self.make_config(ablation_id, variant)
                model = EmbeddingDifferencePGMModel.from_config(config)
                debug = model(self.X, return_debug=True)
                self.assertEqual(tuple(debug["logits"].shape), (self.B, 48))
                if ablation_id == "E0":
                    self.assertIsNone(debug["R"])
                    self.assertEqual(tuple(debug["pooled"].shape), (self.B, 512))
                elif ablation_id in {"E1", "E2"}:
                    self.assertEqual(tuple(debug["R"].shape), (self.B, self.T - 1, 128))
                    self.assertEqual(tuple(debug["Y"].shape), (self.B, self.T - 1, 128))
                    self.assertEqual(tuple(debug["pooled"].shape), (self.B, 128))
                else:
                    self.assertEqual(tuple(debug["R"].shape), (self.B, self.T - 1, 128))
                    self.assertEqual(tuple(debug["Y"].shape), (self.B, self.T - 1, 128))
                    self.assertEqual(tuple(debug["H_final"].shape), (self.B, 8, 128))

    def test_spatial_map_e3_e4_logits_shape(self) -> None:
        spatial_config = load_config(str(PROJECT_ROOT / "configs" / "default_resnet50_spatial_compact.yaml"))
        X = torch.randn(self.B, self.T, 49, 2048)
        for ablation_id, variant in [("E3", "diff_pgm_info"), ("E4", "diff_pgm_info_attention")]:
            with self.subTest(ablation_id=ablation_id, variant=variant):
                config = deepcopy(spatial_config)
                config["model"]["ablation_id"] = ablation_id
                config["model"]["variant"] = variant
                config["classifier"]["type"] = "attention_pool" if ablation_id == "E4" else "mlp"
                model = EmbeddingDifferencePGMModel.from_config(config)
                debug = model(X, return_debug=True)
                self.assertEqual(tuple(debug["logits"].shape), (self.B, 48))
                self.assertEqual(tuple(debug["R"].shape), (self.B, self.T - 1, 64))
                self.assertEqual(tuple(debug["Y"].shape), (self.B, self.T - 1, 64))
                self.assertEqual(tuple(debug["H_final"].shape), (self.B, 8, 64))
                self.assertEqual(tuple(debug["R_tokens"].shape), (self.B, self.T - 1, 49, 64))

    def test_experiment_manager_file_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config("E4", "diff_pgm_info_attention")
            config["output"] = {
                "root": str(Path(tmpdir) / "runs"),
                "registry_path": str(Path(tmpdir) / "experiment_registry.csv"),
            }
            run_paths = create_run_dir(config)
            save_resolved_config(config, run_paths.run_dir)
            write_model_summary(config, run_paths.run_dir)
            init_train_log(run_paths.run_dir)
            append_train_log(
                run_paths.run_dir,
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "val_loss": 1.2,
                    "train_top1": 0.1,
                    "val_top1": 0.2,
                    "lr": 1e-4,
                    "epoch_time_sec": 0.5,
                },
            )
            append_train_log(
                run_paths.run_dir,
                {
                    "epoch": 2,
                    "train_loss": 0.9,
                    "val_loss": 1.1,
                    "train_top1": 0.2,
                    "val_top1": 0.3,
                    "lr": 1e-4,
                    "epoch_time_sec": 0.4,
                },
            )
            metrics = {
                "run_name": run_paths.run_name,
                "date": "2026-05-18T00:00:00",
                "ablation_id": "E4",
                "model_variant": "diff_pgm_info_attention",
                "best_val_top1": 0.3,
                "best_val_epoch": 2,
                "final_train_loss": 0.9,
                "final_val_loss": 1.1,
                "final_train_top1": 0.2,
                "final_val_top1": 0.3,
                "test_loss": None,
                "test_top1": None,
                "lambda_smooth": 1.0,
                "use_alpha": True,
                "classifier_type": "attention_pool",
                "checkpoint_best": str(run_paths.best_checkpoint_path),
                "checkpoint_last": str(run_paths.last_checkpoint_path),
                "notes": "fake test metrics",
            }
            run_paths.best_checkpoint_path.write_bytes(b"fake best checkpoint")
            run_paths.last_checkpoint_path.write_bytes(b"fake last checkpoint")
            save_metrics(run_paths.run_dir, metrics)
            write_experiment_card(config, metrics, run_paths.run_dir)
            append_registry(config, metrics, run_paths.run_dir)

            self.assertTrue(run_paths.config_path.is_file())
            self.assertTrue(run_paths.model_summary_path.is_file())
            self.assertTrue(run_paths.train_log_path.is_file())
            self.assertTrue(run_paths.metrics_path.is_file())
            self.assertTrue(run_paths.experiment_card_path.is_file())
            self.assertTrue(run_paths.best_checkpoint_path.is_file())
            self.assertTrue(run_paths.last_checkpoint_path.is_file())
            self.assertTrue(Path(config["output"]["registry_path"]).is_file())
            with run_paths.train_log_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            data = json.loads(run_paths.metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(data["ablation_id"], "E4")


if __name__ == "__main__":
    unittest.main()
