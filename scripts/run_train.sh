#!/usr/bin/env bash
set -euo pipefail
python -m llm_finetune train --config "${1:-configs/default.yaml}"
