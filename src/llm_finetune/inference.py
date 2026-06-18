from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_finetune.config import AppConfig
from llm_finetune.modeling import ModelFactory
from llm_finetune.data import InstructionRecord, PromptFormatter

class ChatRunner:
    """Interactively test and compare finetuned LoRA adapter over the baseline model."""
    def __init__(self, config: AppConfig, adapter_path: str | Path) -> None:
        self.config = config
        self.adapter_path = str(adapter_path)

    def run(self) -> None:
        print("Loading models for interactive chat... (this may take a moment)")
        # Loading with adapter applied initially
        bundle = ModelFactory(self.config).create_for_evaluation(adapter_path=self.adapter_path)
        model = bundle.model
        tokenizer = bundle.tokenizer
        formatter = PromptFormatter(tokenizer)

        # Default instruction aligned with the jsonl dataset task
        default_instruction = (
            'Classify the user workflow intent and respond with JSON schema: '
            '{"id":"<UUID version 7>","method_to_use":"<describe-globally | describe-with-filter | create | edit | rerun>"}.'
        )

        print("\n" + "=" * 60)
        print(" Interactive Model Comparison: Baseline vs Finetuned")
        print(" Type 'quit' or 'exit' to stop.")
        print("=" * 60)

        while True:
            try:
                user_input = input("\nEnter input query (or 'quit'): ").strip()
                if user_input.lower() in ("quit", "exit"):
                    break
                if not user_input:
                    continue

                # Prepare the structured prompt wrapper
                record = InstructionRecord(
                    id="chat_test",
                    instruction=default_instruction,
                    input=user_input,
                    output=""
                )
                prompt = formatter.format_prompt(record)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                # 1. Generate Fine-tuned Output
                print("\n[🤖 Finetuned Adapter Output]")
                ft_output = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                ft_text = tokenizer.decode(ft_output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                print(f"> {ft_text.strip()}")

                # 2. Generate Baseline Output (Adapter temporarily disabled)
                print("\n[🛑 Baseline Model Output]")
                with model.disable_adapter():
                    base_output = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    base_text = tokenizer.decode(base_output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    print(f"> {base_text.strip()}")

            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat.")
                break
