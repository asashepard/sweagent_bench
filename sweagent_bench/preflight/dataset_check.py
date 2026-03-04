"""Check that the SWE-bench dataset is loadable."""
from __future__ import annotations


def check_dataset(
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
) -> bool:
    """Try loading the dataset and verify it has instances.

    Args:
        dataset_name: HuggingFace dataset name.
        split: Dataset split to load.

    Returns:
        True if dataset loads and contains at least one instance.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split)
        count = len(ds)
        if count > 0:
            print(f"    Dataset: {count} instances in {dataset_name}/{split}")
            return True
        print(f"    Dataset loaded but empty: {dataset_name}/{split}")
        return False
    except ImportError:
        print("    'datasets' package not installed")
        return False
    except Exception as exc:
        print(f"    Dataset load error: {exc}")
        return False
