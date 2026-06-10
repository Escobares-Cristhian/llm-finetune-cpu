from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import EvalPrediction


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    del labels
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


class CausalLMMetrics:
    @staticmethod
    def compute(eval_prediction: EvalPrediction) -> dict[str, float]:
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        preds = np.asarray(predictions)
        label_ids = np.asarray(labels)

        # For causal LM, position t predicts t+1.
        shifted_preds = preds[:, :-1]
        shifted_labels = label_ids[:, 1:]
        mask = shifted_labels != -100

        if mask.sum() == 0:
            return {"token_accuracy": 0.0}

        token_accuracy = (shifted_preds[mask] == shifted_labels[mask]).mean().item()
        return {"token_accuracy": float(token_accuracy)}


class MetricsStore:
    def __init__(self, metrics_dir: Path) -> None:
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, payload: dict[str, Any] | list[dict[str, Any]]) -> Path:
        path = self.metrics_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @staticmethod
    def add_perplexity(metrics: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(metrics)
        for key, value in list(metrics.items()):
            if key.endswith("loss") and isinstance(value, int | float):
                perplexity_key = key.replace("loss", "perplexity")
                enriched[perplexity_key] = safe_perplexity(float(value))
        return enriched


def safe_perplexity(loss: float) -> float:
    if loss > 20:
        return float("inf")
    return float(math.exp(loss))


class MetricsPlotter:
    def __init__(self, report_dir: Path) -> None:
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def plot_all(self, log_history: list[dict[str, Any]], summary: dict[str, Any]) -> list[Path]:
        paths = [
            self.plot_training_loss(log_history),
            self.plot_eval_loss(log_history),
            self.plot_improvement(summary, metric_suffix="loss", filename="loss_improvement.png"),
            self.plot_improvement(
                summary,
                metric_suffix="token_accuracy",
                filename="token_accuracy_improvement.png",
            ),
        ]
        return [path for path in paths if path is not None]

    def plot_training_loss(self, log_history: list[dict[str, Any]]) -> Path | None:
        rows = [row for row in log_history if "loss" in row and "step" in row]
        if not rows:
            return None
        return self._line_plot(
            x=[row["step"] for row in rows],
            y=[row["loss"] for row in rows],
            title="Training loss by step",
            xlabel="Step",
            ylabel="Loss",
            filename="training_loss.png",
        )

    def plot_eval_loss(self, log_history: list[dict[str, Any]]) -> Path | None:
        rows = [row for row in log_history if "eval_loss" in row and "step" in row]
        if not rows:
            return None
        return self._line_plot(
            x=[row["step"] for row in rows],
            y=[row["eval_loss"] for row in rows],
            title="Validation loss by step",
            xlabel="Step",
            ylabel="Validation loss",
            filename="eval_loss.png",
        )

    def plot_improvement(
        self,
        summary: dict[str, Any],
        metric_suffix: str,
        filename: str,
    ) -> Path | None:
        pairs = [
            ("val", f"baseline_val_{metric_suffix}", f"final_val_{metric_suffix}"),
            ("test", f"baseline_test_{metric_suffix}", f"final_test_{metric_suffix}"),
            ("hard", f"baseline_hard_{metric_suffix}", f"final_hard_{metric_suffix}"),
        ]
        labels: list[str] = []
        baseline: list[float] = []
        final: list[float] = []
        for label, before_key, after_key in pairs:
            if before_key in summary and after_key in summary:
                labels.append(label)
                baseline.append(float(summary[before_key]))
                final.append(float(summary[after_key]))

        if not labels:
            return None

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, baseline, width, label="baseline")
        ax.bar(x + width / 2, final, width, label="final")
        ax.set_title(metric_suffix.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric_suffix.replace("_", " "))
        ax.legend()
        fig.tight_layout()
        path = self.report_dir / filename
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    def _line_plot(
        self,
        x: list[float],
        y: list[float],
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, marker="o")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        path = self.report_dir / filename
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path
