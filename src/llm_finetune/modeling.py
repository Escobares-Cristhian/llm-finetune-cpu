from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from llm_finetune.config import AppConfig

# Force CPU even if the host has a GPU exposed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@dataclass
class ModelBundle:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


class ModelFactory:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def create_for_training(self) -> ModelBundle:
        tokenizer = self._load_tokenizer()
        model = self._load_base_model()
        model = self._apply_lora(model)
        model.print_trainable_parameters()
        return ModelBundle(model=model, tokenizer=tokenizer)

    def create_for_evaluation(self, adapter_path: str | None = None) -> ModelBundle:
        tokenizer = self._load_tokenizer()
        model = self._load_base_model()
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        return ModelBundle(model=model, tokenizer=tokenizer)

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_id,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _load_base_model(self) -> PreTrainedModel:
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.model.trust_remote_code,
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
        }
        if self.config.model.attn_implementation:
            kwargs["attn_implementation"] = self.config.model.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(self.config.model.model_id, **kwargs)
        model.to("cpu")
        model.config.use_cache = False
        return model

    def _apply_lora(self, model: PreTrainedModel) -> PreTrainedModel:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            target_modules=self.config.lora.target_modules,
        )
        return get_peft_model(model, lora_config)
