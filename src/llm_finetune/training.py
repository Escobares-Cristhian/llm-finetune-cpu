from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from transformers import Trainer, TrainingArguments, set_seed, TrainerCallback

from llm_finetune.config import AppConfig
from llm_finetune.data import (
    CausalCollator,
    DatasetSplits,
    InstructionTuningDataset,
    JsonlDataRepository,
)
from llm_finetune.metrics import (
    CausalLMMetrics,
    MetricsPlotter,
    MetricsStore,
    preprocess_logits_for_metrics,
)
from llm_finetune.modeling import ModelFactory

# CPU-only guard before Trainer/Accelerate initialize devices.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


class PlottingCallback(TrainerCallback):
    """Callback to update training and evaluation plots progressively during training."""
    def __init__(self, report_dir: Path) -> None:
        self.plotter = MetricsPlotter(report_dir)

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: dict[str, float] | None = None, **kwargs: Any) -> None:
        self.plotter.plot_training_loss(state.log_history)
        self.plotter.plot_eval_loss(state.log_history)


class FineTuningRunner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.ensure_dirs()
        set_seed(self.config.training.seed)

    def run(self) -> dict[str, Any]:
        bundle = ModelFactory(self.config).create_for_training()
        splits = JsonlDataRepository(self.config.data).load_splits()
        datasets = self._build_datasets(splits, bundle.tokenizer)

        trainer = self._build_trainer(bundle.model, bundle.tokenizer, datasets)
        metrics_store = MetricsStore(self.config.training.metrics_dir)

        summary: dict[str, Any] = {}
        summary.update(self._evaluate_all(trainer, datasets, prefix="baseline"))

        trainer.train()
        trainer.save_model(str(self.config.training.final_adapter_dir))
        bundle.tokenizer.save_pretrained(str(self.config.training.final_adapter_dir))

        summary.update(self._evaluate_all(trainer, datasets, prefix="final"))
        summary = MetricsStore.add_perplexity(summary)

        metrics_store.save_json("eval_summary.json", summary)
        log_history = trainer.state.log_history
        metrics_store.save_json("log_history.json", log_history)

        plot_paths = MetricsPlotter(self.config.training.report_dir).plot_all(log_history, summary)
        summary["plots"] = [str(path) for path in plot_paths]
        metrics_store.save_json("eval_summary.json", summary)
        return summary

    def _build_datasets(self, splits: DatasetSplits, tokenizer: Any) -> dict[str, InstructionTuningDataset]:
        max_length = self.config.model.max_length
        return {
            "train": InstructionTuningDataset(splits.train, tokenizer, max_length),
            "val": InstructionTuningDataset(splits.val, tokenizer, max_length),
            "test": InstructionTuningDataset(splits.test, tokenizer, max_length),
            "hard": InstructionTuningDataset(splits.hard_test, tokenizer, max_length),
        }

    def _build_trainer(self, model: Any, tokenizer: Any, datasets: dict[str, Any]) -> Trainer:
        args = TrainingArguments(
            output_dir=str(self.config.training.output_dir),
            num_train_epochs=self.config.training.num_train_epochs,
            max_steps=self.config.training.max_steps,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            logging_strategy="steps",
            logging_steps=self.config.training.logging_steps,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=self.config.training.eval_steps,
            save_strategy="steps",
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            report_to="none",
            fp16=False,
            bf16=False,
            optim="adamw_torch",
            remove_unused_columns=False,
            seed=self.config.training.seed,
        )
        return Trainer(
            model=model,
            args=args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
            data_collator=CausalCollator(tokenizer),
            compute_metrics=CausalLMMetrics.compute,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[PlottingCallback(self.config.training.report_dir)],
        )

    @staticmethod
    def _evaluate_all(trainer: Trainer, datasets: dict[str, Any], prefix: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for name in ["val", "test", "hard"]:
            result = trainer.evaluate(eval_dataset=datasets[name], metric_key_prefix=f"{prefix}_{name}")
            metrics.update(result)
        return metrics


class EvaluationRunner:
    def __init__(self, config: AppConfig, adapter_path: str | Path) -> None:
        self.config = config
        self.adapter_path = str(adapter_path)
        self.config.ensure_dirs()

    def run(self) -> dict[str, Any]:
        bundle = ModelFactory(self.config).create_for_evaluation(adapter_path=self.adapter_path)
        splits = JsonlDataRepository(self.config.data).load_splits()
        datasets = {
            "test": InstructionTuningDataset(splits.test, bundle.tokenizer, self.config.model.max_length),
            "hard": InstructionTuningDataset(splits.hard_test, bundle.tokenizer, self.config.model.max_length),
        }
        args = TrainingArguments(
            output_dir=str(self.config.training.output_dir),
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            report_to="none",
            fp16=False,
            bf16=False,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=bundle.model,
            args=args,
            data_collator=CausalCollator(bundle.tokenizer),
            compute_metrics=CausalLMMetrics.compute,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        summary: dict[str, Any] = {}
        for name, dataset in datasets.items():
            summary.update(trainer.evaluate(eval_dataset=dataset, metric_key_prefix=name))
        summary = MetricsStore.add_perplexity(summary)
        MetricsStore(self.config.training.metrics_dir).save_json("standalone_eval_summary.json", summary)
        return summary
