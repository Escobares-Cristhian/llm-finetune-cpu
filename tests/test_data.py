from llm_finetune.config import DataConfig
from llm_finetune.data import InstructionRecord, JsonlDataRepository, RecordSplitter, write_jsonl


def make_records(n: int) -> list[InstructionRecord]:
    return [
        InstructionRecord(
            id=f"r-{i}",
            instruction="Do something",
            input=f"input {i}",
            output=f"output {i}",
            source="test",
        )
        for i in range(n)
    ]


def test_splitter_is_deterministic() -> None:
    records = make_records(10)
    splitter = RecordSplitter(seed=123, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    first = splitter.split(records)
    second = splitter.split(records)

    assert [[row.id for row in part] for part in first] == [[row.id for row in part] for part in second]


def test_splitter_keeps_validation_and_test_for_small_data() -> None:
    records = make_records(3)
    train, val, test = RecordSplitter(seed=1, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1).split(records)

    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1


def test_repository_applies_max_examples_before_splitting(tmp_path) -> None:
    manual_path = tmp_path / "manual.jsonl"
    augmented_path = tmp_path / "augmented.jsonl"
    hard_path = tmp_path / "hard.jsonl"

    write_jsonl(manual_path, make_records(5))
    write_jsonl(augmented_path, make_records(10))
    write_jsonl(hard_path, make_records(6))

    config = DataConfig(
        manual_path=manual_path,
        augmented_path=augmented_path,
        hard_test_path=hard_path,
        processed_dir=tmp_path / "processed",
        max_examples=3,
        seed=1,
    )

    splits = JsonlDataRepository(config).load_splits()

    assert len(splits.train) == 4  # 3 manual + 1 augmented train row
    assert len(splits.val) == 1
    assert len(splits.test) == 1
    assert len(splits.hard_test) == 3
