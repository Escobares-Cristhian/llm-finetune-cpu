from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_finetune.config import AppConfig
from llm_finetune.training import EvaluationRunner, FineTuningRunner
from llm_finetune.inference import ChatRunner


class CliApp:
    def __init__(self) -> None:
        self.parser = self._build_parser()

    def run(self) -> None:
        args = self.parser.parse_args()
        if args.command == "train":
            config = AppConfig.from_yaml(args.config)
            summary = FineTuningRunner(config).run()
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif args.command == "evaluate":
            config = AppConfig.from_yaml(args.config)
            summary = EvaluationRunner(config, args.adapter).run()
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif args.command == "chat":
            config = AppConfig.from_yaml(args.config)
            ChatRunner(config, args.adapter).run()
        else:
            self.parser.print_help()

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="CPU-only LoRA fine-tuning for causal LMs")
        subparsers = parser.add_subparsers(dest="command")

        train = subparsers.add_parser("train", help="Train LoRA adapter and run final metrics")
        train.add_argument("--config", type=Path, required=True)

        evaluate = subparsers.add_parser("evaluate", help="Evaluate an existing LoRA adapter")
        chat = subparsers.add_parser("chat", help="Interactive chat to compare baseline and finetuned models")
        chat.add_argument("--config", type=Path, required=True)
        chat.add_argument("--adapter", type=Path, required=True)

        evaluate.add_argument("--config", type=Path, required=True)
        evaluate.add_argument("--adapter", type=Path, required=True)

        return parser


def main() -> None:
    CliApp().run()
