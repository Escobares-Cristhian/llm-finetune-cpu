from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from llm_finetune.config import DataConfig


REQUIRED_FIELDS = {"instruction", "output"}


@dataclass(frozen=True)
class InstructionRecord:
    id: str
    instruction: str
    input: str
    output: str
    source: str = "unknown"

    @classmethod
    def from_dict(cls, row: dict[str, Any], fallback_id: str) -> "InstructionRecord":
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"Record {fallback_id} missing fields: {sorted(missing)}")
        return cls(
            id=str(row.get("id", fallback_id)),
            instruction=str(row["instruction"]).strip(),
            input=str(row.get("input", "")).strip(),
            output=str(row["output"]).strip(),
            source=str(row.get("source", "unknown")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "source": self.source,
        }


@dataclass(frozen=True)
class DatasetSplits:
    train: list[InstructionRecord]
    val: list[InstructionRecord]
    test: list[InstructionRecord]
    hard_test: list[InstructionRecord]


class JsonlDataRepository:
    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def load_splits(self) -> DatasetSplits:
        manual = self._limit_records(self._read_jsonl(self.config.manual_path, "manual"))
        augmented = self._limit_records(self._read_jsonl(self.config.augmented_path, "augmented"))
        hard_test = self._limit_records(self._read_jsonl(self.config.hard_test_path, "hard_test"))

        aug_train, aug_val, aug_test = RecordSplitter(
            seed=self.config.seed,
            train_ratio=self.config.augmented_train_ratio,
            val_ratio=self.config.augmented_val_ratio,
            test_ratio=self.config.augmented_test_ratio,
        ).split(augmented)

        train = [*aug_train]
        if self.config.include_manual_in_train:
            train = [*manual, *train]

        splits = DatasetSplits(train=train, val=aug_val, test=aug_test, hard_test=hard_test)
        self._write_processed_splits(splits)
        return splits

    def _limit_records(self, records: list[InstructionRecord]) -> list[InstructionRecord]:
        if self.config.max_examples is None:
            return records
        return records[: self.config.max_examples]

    @staticmethod
    def _read_jsonl(path: Path, fallback_source: str) -> list[InstructionRecord]:
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")

        records: list[InstructionRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                row.setdefault("source", fallback_source)
                records.append(InstructionRecord.from_dict(row, f"{path.name}:{line_number}"))
        if not records:
            raise ValueError(f"Data file has no records: {path}")
        return records

    def _write_processed_splits(self, splits: DatasetSplits) -> None:
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        for name, records in {
            "train": splits.train,
            "val": splits.val,
            "test": splits.test,
            "hard_test": splits.hard_test,
        }.items():
            write_jsonl(self.config.processed_dir / f"{name}.jsonl", records)


class RecordSplitter:
    def __init__(self, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0; got {total:.4f}")
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(
        self, records: list[InstructionRecord]
    ) -> tuple[list[InstructionRecord], list[InstructionRecord], list[InstructionRecord]]:
        shuffled = list(records)
        random.Random(self.seed).shuffle(shuffled)

        n = len(shuffled)
        train_end = max(1, int(n * self.train_ratio))
        val_size = max(1, int(n * self.val_ratio)) if n >= 3 else 0
        val_end = min(n, train_end + val_size)

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        if n >= 3 and not test:
            test = [train.pop()]
        if n >= 3 and not val:
            val = [train.pop()]

        return train, val, test


class PromptFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def format_prompt(self, record: InstructionRecord) -> str:
        user_content = record.instruction
        if record.input:
            user_content = f"{record.instruction}\n\nInput:\n{record.input}"
        messages = [{"role": "user", "content": user_content}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"### Instruction\n{user_content}\n\n### Response\n"

    def format_full_text(self, record: InstructionRecord) -> str:
        user_content = record.instruction
        if record.input:
            user_content = f"{record.instruction}\n\nInput:\n{record.input}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record.output},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        return f"### Instruction\n{user_content}\n\n### Response\n{record.output}{self.tokenizer.eos_token or ''}"


class InstructionTuningDataset(Dataset):
    def __init__(
        self,
        records: list[InstructionRecord],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.formatter = PromptFormatter(tokenizer)
        self.features = [self._tokenize(record) for record in records]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in self.features[index].items()
        }

    def _tokenize(self, record: InstructionRecord) -> dict[str, list[int]]:
        prompt = self.formatter.format_prompt(record)
        full_text = self.formatter.format_full_text(record)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        tokenized = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids: list[int] = tokenized["input_ids"]
        attention_mask: list[int] = tokenized["attention_mask"]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class CausalCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_len - label.shape[0]
            if pad_len > 0:
                label = torch.cat([label, torch.full((pad_len,), -100, dtype=torch.long)])
            padded_labels.append(label)
        batch["labels"] = torch.stack(padded_labels)
        return batch


def write_jsonl(path: Path, records: Iterable[InstructionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
