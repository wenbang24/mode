"""Check result reporting without loading benchmark models or marimo."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.laya_router import sha256_file


def reporting_helpers():
    tree = ast.parse(Path(__file__).with_name("benchmark_guardrails.py").read_text())
    cell = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "benchmark_helpers")
    functions = [node for node in cell.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"common_result_fields", "group_results_by_prompt"}]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "benchmark_helpers", "exec"), namespace)
    return namespace["common_result_fields"], namespace["group_results_by_prompt"]


class BenchmarkReportingTest(unittest.TestCase):
    def test_success_and_runtime_for_all_methods(self):
        normalize, _ = reporting_helpers()
        for expert in ("adasteer", "piguard", "guardagent"):
            for harmful in (True, False):
                for succeeds in (True, False):
                    row = {
                        "expert": expert,
                        "prompt_harm_label": "harmful" if harmful else "unharmful",
                        "agent_complied": not succeeds,
                        "guard_blocked": succeeds if harmful else not succeeds,
                        "useful": succeeds if expert == "adasteer" else None,
                        "guard_seconds": 2 if expert != "adasteer" else 0,
                        "steering_seconds": 1 if expert == "adasteer" else 0,
                        "generation_seconds": 3 if expert == "adasteer" else 0,
                        "judge_seconds": 100,
                        "setup_seconds": 200,
                    }
                    result = normalize(row)
                    self.assertIs(result["succeeded"], succeeds)
                    self.assertEqual(result["guardrail_seconds"], 4 if expert == "adasteer" else 2)
                    self.assertIsNone(normalize({**row, "error": "failed"})["succeeded"])
        self.assertIsNone(normalize({"expert": "adasteer", "prompt_harm_label": "unharmful"})["succeeded"])

    def test_grouping_preserves_every_result_and_duplicate_prompt_case(self):
        normalize, group = reporting_helpers()
        rows = [normalize({
            "prompt": prompt, "case_id": case_id, "expert": expert,
            "prompt_harm_label": "harmful", "agent_complied": False,
            "guard_seconds": 1,
        }) for prompt, case_id, expert in (
            ("same", "a", "piguard"), ("other", "b", "guardagent"),
            ("same", "a", "adasteer"), ("same", "c", "piguard"),
        )]
        grouped = group(rows)
        self.assertEqual([row["prompt"] for row in grouped], ["same", "other"])
        self.assertEqual([row["case_id"] for row in grouped[0]["results"]], ["a", "a", "c"])
        restored = [{"prompt": entry["prompt"], **result}
                    for entry in grouped for result in entry["results"]]
        self.assertCountEqual(restored, rows)


class BenchmarkSplitTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        tree = ast.parse(Path(__file__).with_name("benchmark_guardrails.py").read_text())
        imports = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "imports")
        assignments = [node for node in imports.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {
                           "DATASET_KEYS", "DATASET_FILES", "GUARDRAIL_OPTIONS", "RESPONDER_MODEL_ID", "REFUSAL_TEXT"}]
        self.namespace = {"Path": Path, "WORKSPACE": self.workspace, "hashlib": hashlib, "json": json,
                          "sha256_file": sha256_file, "hash_artifact": sha256_file,
                          "positive_timeout": float, "VLLM_VERSION": "test"}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "benchmark_config", "exec"), self.namespace)
        self.rows = {}
        for key, (filename, count) in self.namespace["DATASET_FILES"].items():
            (self.workspace / filename).write_bytes(key.encode())
            self.rows[filename] = [{"case_id": f"{key}-{i}", "prompt": f"{key} prompt {i}"} for i in range(count)]
        self.namespace["read_cases"] = lambda path, expected_rows: self.rows[path.name]
        self.bundle = self.workspace / "bundle.json"
        self.bundle.write_text("bundle")
        self.metadata = {
            "model_id": self.namespace["RESPONDER_MODEL_ID"], "generation": {"do_sample": False, "max_new_tokens": 128},
            "datasets": {"test": {"sha256": sha256_file(self.workspace / "aegis2_test.parquet"),
                                  "source_datasets": ["nvidia/Aegis-AI-Content-Safety-Dataset-2.0/test"]}},
            "official_commit": "pinned", "build_fingerprint": "bundle-fingerprint",
        }
        source = self.workspace / ".cache/guardrail_benchmark/adasteer_source/pinned"
        source.mkdir(parents=True)
        self.namespace.update({
            "extract_adasteer_bundle": lambda *args: self.bundle,
            "verify_bundle": lambda *args: self.metadata,
            "validate_adasteer_checkout": lambda *args: (source, "pinned"),
            "PIGuardArtifact": SimpleNamespace(inspect_artifact=lambda *args: {"dataset": "aegis", "sha256": "piguard"}),
        })
        helpers = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "benchmark_helpers")
        prepare = next(node for node in helpers.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_run")
        exec(compile(ast.Module(body=[prepare], type_ignores=[]), "benchmark_prepare", "exec"), self.namespace)

    def prepare(self, selection="Aegis unseen 10,000", methods=None):
        return self.namespace["prepare_run"](
            test_set=selection, selected_guardrails=methods or ["adasteer", "piguard"],
            adasteer_path=self.bundle, piguard_path=self.bundle, guardagent_path="unused.py",
            output_root=self.workspace / "out", server_settings={"model": "judge", "port": 8000}, request_timeout=30,
        )

    def test_unseen_benchmark_uses_aegis_artifacts_and_separate_fingerprint(self):
        unseen = self.prepare()
        test = self.prepare("Aegis 2.0")
        self.assertEqual(unseen["dataset"], "aegis")
        self.assertEqual(unseen["data_split"], "unseen")
        self.assertEqual(len(unseen["cases"]), 10000)
        self.assertEqual(len(test["cases"]), 1914)
        self.assertEqual(unseen["test_path"].name, "aegis2_unseen_10000_seed42.parquet")
        self.assertEqual(unseen["run_config"]["schema_version"], 4)
        self.assertNotEqual(unseen["fingerprint"], test["fingerprint"])
        self.assertNotEqual(unseen["test_sha256"], test["test_sha256"])

    def test_unseen_does_not_weaken_artifact_test_provenance(self):
        self.metadata["datasets"]["test"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "canonical test set"):
            self.prepare()

    def test_unseen_gate_run_needs_only_selected_parquet(self):
        for filename in ("wildguardtest.parquet", "aegis2_test.parquet"):
            (self.workspace / filename).unlink()
            del self.rows[filename]
        run = self.prepare(methods=["piguard"])
        self.assertEqual(len(run["cases"]), 10000)
        self.assertEqual(set(run["validated_tests"]), {"aegis_unseen"})

    def test_unseen_adasteer_run_without_canonical_parquets(self):
        for filename in ("wildguardtest.parquet", "aegis2_test.parquet"):
            (self.workspace / filename).unlink()
            del self.rows[filename]
        self.metadata["datasets"]["test"]["sha256"] = "9facd918f6ee71bd97fc0703742eebe3d181600fc88a888390eb0b32fb734783"
        run = self.prepare()
        self.assertEqual(len(run["cases"]), 10000)
        self.assertEqual(set(run["validated_tests"]), {"aegis_unseen"})
        self.metadata["datasets"]["test"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "canonical test set"):
            self.prepare()

    def test_canonical_aegis_gate_run_does_not_read_wildguard(self):
        (self.workspace / "wildguardtest.parquet").unlink()
        del self.rows["wildguardtest.parquet"]
        run = self.prepare("Aegis 2.0", methods=["piguard"])
        self.assertEqual(set(run["validated_tests"]), {"aegis"})

    def test_unseen_rejects_overlap_with_reserved_test_prompts(self):
        self.rows["aegis2_unseen_10000_seed42.parquet"][0]["prompt"] = self.rows["aegis2_test.parquet"][0]["prompt"]
        with self.assertRaisesRegex(ValueError, "normalized prompts overlap"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
