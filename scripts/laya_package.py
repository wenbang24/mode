"""Package validated router-training inputs without model weights or credentials."""
import io
from pathlib import Path
import zipfile

from scripts.laya_router import assert_disjoint, load_benchmark
from scripts.training_cases import read_cases


def finetuner_files(workspace, run_directory):
    workspace, run_directory = Path(workspace), Path(run_directory)
    unseen = workspace / "aegis2_unseen_1000_seed42.parquet"
    reserved = workspace / "aegis2_test.parquet"
    data = load_benchmark(run_directory, unseen, expected_rows=1000, split="unseen")
    assert_disjoint(data.cases, read_cases(reserved, expected_rows=1914))
    files = {
        name: workspace / name for name in (
            "train_laya_router.py", "scripts/laya_router.py", "scripts/training_cases.py",
            unseen.name, reserved.name,
        )
    }
    files["benchmarks/unseen/run_config.json"] = run_directory / "run_config.json"
    files[f"benchmarks/unseen/{data.results_path.name}"] = data.results_path
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required finetuner file is missing: {path}")
    return files


def build_finetuner_zip(workspace, run_directory):
    files = finetuner_files(workspace, run_directory)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, arcname=name)
        archive.writestr("README.txt", """MoDE Laya finetuner

Extract this ZIP, keeping its directory structure. Open train_laya_router.py
in Molab, or locally with:
uvx marimo@0.24.0 edit train_laya_router.py --sandbox

The bundled benchmark is in benchmarks/unseen and is selected automatically.
Click Validate benchmark and prepare training data, then Download / reuse
pinned Laya checkpoint. The notebook installs its declared dependencies and
downloads the model separately; internet access is needed for those steps.

This package contains training supervision and the reserved test prompts.
It does not contain a final test benchmark, expert models, or credentials.
Fine-tuning, validation, and export do not require a final test benchmark.
""")
    return output.getvalue()
