from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    trust_remote_code: bool = True
    max_length: int = 512
    attn_implementation: str | None = None


@dataclass(frozen=True)
class LoraConfigData:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] | None = None


@dataclass(frozen=True)
class DataConfig:
    manual_path: Path
    augmented_path: Path
    hard_test_path: Path
    processed_dir: Path
    include_manual_in_train: bool = True
    augmented_train_ratio: float = 0.8
    augmented_val_ratio: float = 0.1
    augmented_test_ratio: float = 0.1
    seed: int = 42
    max_examples: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    final_adapter_dir: Path
    metrics_dir: Path
    report_dir: Path
    num_train_epochs: float = 2.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 1
    eval_steps: int = 10
    save_steps: int = 10
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    seed: int = 42


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    lora: LoraConfigData
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        load_dotenv()
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a mapping: {config_path}")

        data_values = _paths(
            raw["data"],
            ["manual_path", "augmented_path", "hard_test_path", "processed_dir"],
        )
        data_values["max_examples"] = read_positive_int_env("MAX_EXAMPLES")

        return cls(
            model=ModelConfig(**raw["model"]),
            lora=LoraConfigData(**raw["lora"]),
            data=DataConfig(**data_values),
            training=TrainingConfig(
                **_paths(
                    raw["training"],
                    ["output_dir", "final_adapter_dir", "metrics_dir", "report_dir"],
                )
            ),
        )

    def ensure_dirs(self) -> None:
        self.data.processed_dir.mkdir(parents=True, exist_ok=True)
        self.training.output_dir.mkdir(parents=True, exist_ok=True)
        self.training.final_adapter_dir.mkdir(parents=True, exist_ok=True)
        self.training.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.training.report_dir.mkdir(parents=True, exist_ok=True)


def _paths(values: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    copied = dict(values)
    for key in keys:
        copied[key] = Path(copied[key])
    return copied


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file without extra dependencies."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('\"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_positive_int_env(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if raw in {"", "0", "none", "None", "null", "NULL"}:
        return None

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer or empty; got {raw!r}") from exc

    if value < 1:
        raise ValueError(f"{name} must be a positive integer or empty; got {value}")
    return value
