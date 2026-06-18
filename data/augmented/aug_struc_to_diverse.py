import json
import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm


# =========================
# Config
# =========================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "augmented_structured.jsonl"
OUTPUT_FILE = BASE_DIR / "augmented_structured_lexic_changed.jsonl"

MODEL = "openai/gpt-oss-120b"

INPUT_KEY = "input"

MAX_WORKERS = 8          # Increase if your Groq rate limits allow it
BATCH_SIZE = 32          # Number of JSONL rows handled per batch
MAX_RETRIES = 4

TEMPERATURE = 0.7
MAX_COMPLETION_TOKENS = 180

RESUME = True


# =========================
# Setup
# =========================

load_dotenv(BASE_DIR / ".env")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Missing GROQ_API_KEY in .env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================
# Helpers
# =========================

def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def extract_json(text: str) -> dict | None:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def make_messages(text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You rewrite dataset inputs. "
                "Keep the exact same meaning, intent, entities, constraints, and language. "
                "Change only the wording. "
                "Do not add details. Do not remove details. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": f"""
Rewrite this text using different wording.

Rules:
- Same exact meaning and intent.
- Same language.
- Do not answer the instruction.
- Do not explain.
- Do not change names, numbers, dates, file names, URLs, code, variables, or quoted text.
- Return only this JSON:

{{"rewrite": "..."}}

Text:
{text}
""".strip(),
        },
    ]


def rewrite_input(text: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=make_messages(text),
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                response_format={"type": "json_object"},
                include_reasoning=False,
                reasoning_effort="low",
            )

            content = response.choices[0].message.content or ""
            parsed = extract_json(content)

            if isinstance(parsed, dict):
                rewritten = parsed.get("rewrite")
                if isinstance(rewritten, str) and rewritten.strip():
                    return rewritten.strip()

        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"Request failed on attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(wait)

    return text


def process_object(obj: dict) -> dict:
    if INPUT_KEY in obj and isinstance(obj[INPUT_KEY], str):
        obj[INPUT_KEY] = rewrite_input(obj[INPUT_KEY])
    return obj


def batched(iterator, size: int):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_jsonl_from(path: Path, skip_lines: int = 0):
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < skip_lines:
                continue

            line = line.rstrip("\n")

            if not line.strip():
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Invalid JSON at line {line_idx + 1}; preserving raw line.")
                yield line


# =========================
# Main
# =========================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    total_lines = count_lines(INPUT_FILE)
    done_lines = count_lines(OUTPUT_FILE) if RESUME else 0

    if done_lines > total_lines:
        raise RuntimeError(
            "Output has more lines than input. Delete the output file or set RESUME = False."
        )

    mode = "a" if RESUME and done_lines > 0 else "w"

    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Model:  {MODEL}")
    print(f"Total lines: {total_lines}")
    print(f"Already done: {done_lines}")
    print(f"Workers: {MAX_WORKERS}")
    print()

    rows = read_jsonl_from(INPUT_FILE, skip_lines=done_lines)

    with OUTPUT_FILE.open(mode, encoding="utf-8") as out:
        with tqdm(total=total_lines, initial=done_lines, desc="Processing") as pbar:
            for batch in batched(rows, BATCH_SIZE):
                results = [None] * len(batch)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_idx = {}

                    for idx, item in enumerate(batch):
                        if isinstance(item, dict):
                            future = executor.submit(process_object, item)
                            future_to_idx[future] = idx
                        else:
                            results[idx] = item

                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]

                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            print(f"Unexpected failure in row: {e}")
                            results[idx] = batch[idx]

                for item in results:
                    if isinstance(item, dict):
                        out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    else:
                        out.write(str(item) + "\n")

                out.flush()
                pbar.update(len(batch))

    print()
    print("Done.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
