.PHONY: build train evaluate test lint

build:
	docker build -t llm-finetune-cpu .

train:
	docker run --rm -it --env-file .env -v "$$(pwd)/.cache:/app/.cache" -e HF_HOME=/app/.cache/huggingface -v "$$(pwd)/data:/app/data" -v "$$(pwd)/outputs:/app/outputs" -v "$$(pwd)/reports:/app/reports" llm-finetune-cpu train --config configs/default.yaml

evaluate:
	docker run --rm -it --env-file .env -v "$$(pwd)/.cache:/app/.cache" -e HF_HOME=/app/.cache/huggingface -v "$$(pwd)/data:/app/data" -v "$$(pwd)/outputs:/app/outputs" -v "$$(pwd)/reports:/app/reports" llm-finetune-cpu evaluate --config configs/default.yaml --adapter outputs/final_adapter

test:
	pytest

lint:
	ruff check src tests
