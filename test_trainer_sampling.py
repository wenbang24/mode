"""Test-sample limits shared by the three trainer workflows."""
import unittest
from scripts.training_cases import sample_test_cases


class TestSamplingTest(unittest.TestCase):
    def test_limits_and_reproducibility(self):
        rows = [{"case_id": str(i), "prompt_harm_label": "harmful" if i < 3 else "unharmful"}
                for i in range(20)]
        self.assertEqual(sample_test_cases(rows), rows)
        self.assertEqual(sample_test_cases(rows, 100), rows)
        for limit in (2, 3, 5, 19):
            selected = sample_test_cases(rows, limit)
            self.assertEqual(len(selected), limit)
            self.assertEqual(selected, sample_test_cases(rows, limit))
            self.assertEqual({r["prompt_harm_label"] for r in selected}, {"harmful", "unharmful"})
            self.assertEqual(len({r["case_id"] for r in selected}), limit)
        for invalid in (-1, 1, 2.5, True, None):
            with self.assertRaises(ValueError):
                sample_test_cases(rows, invalid)

    def test_adasteer_loader_samples_only_test(self):
        from pathlib import Path
        from unittest.mock import patch
        from scripts.experts import adasteer_bundle as bundle
        splits = {
            name: [{"case_id": f"{name}:{i}", "source_dataset": "fixture", "source_index": i,
                    "prompt": f"{name} prompt {i}", "prompt_harm_label": "harmful" if i % 2 else "unharmful",
                    "adversarial": False, "subcategory": "fixture"} for i in range(10)]
            for name in ("train", "validation", "test")
        }
        with patch.object(bundle, "read_cases", side_effect=lambda path, **kw: splits[path.name]):
            _, rows = bundle.load_dataset_splits(
                Path("train"), Path("validation"), Path("test"), train_size=6, validation_size=4,
                test_case_limit=2,
            )
        self.assertEqual({k: len(v) for k, v in rows.items()}, {"train": 6, "validation": 4, "test": 2})
        self.assertEqual([r["case_id"] for r in rows["test"]],
                         [r["case_id"] for r in sample_test_cases(splits["test"], 2)])
