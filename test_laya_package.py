import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import laya_package as package
from scripts import laya_router as router
from test_laya_router import make_benchmark, make_cases, write_results


class FinetunerPackageTest(unittest.TestCase):
    def test_portable_archive_and_incomplete_run_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            for name in ("train_laya_router.py", "scripts/laya_router.py", "scripts/training_cases.py",
                         "aegis2_unseen_1000_seed42.parquet", "aegis2_test.parquet"):
                (root / name).write_bytes(name.encode())
            cases = make_cases(1000)
            run = root / "run"
            rows = make_benchmark(run, cases, root / "aegis2_unseen_1000_seed42.parquet")
            with patch.object(router, "read_cases", return_value=cases), patch.object(
                package, "read_cases", return_value=make_cases(1914, "test")
            ):
                content = package.build_finetuner_zip(root, run)
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    self.assertIsNone(archive.testzip())
                    archive.extractall(root / "extracted")
                    self.assertIn("benchmarks/unseen/results.jsonl", archive.namelist())
                    self.assertEqual(archive.read("aegis2_test.parquet"), b"aegis2_test.parquet")
                # The extracted run remains valid with the new relative paths.
                restored = router.load_benchmark(
                    root / "extracted/benchmarks/unseen",
                    root / "extracted/aegis2_unseen_1000_seed42.parquet",
                    expected_rows=1000, split="unseen",
                )
                self.assertEqual(len(restored.outcomes), 1000)
                write_results(run, rows[:-1])
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    package.build_finetuner_zip(root, run)


if __name__ == "__main__":
    unittest.main()
