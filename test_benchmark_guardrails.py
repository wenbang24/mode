"""Check result reporting without loading benchmark models or marimo."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
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
    def test_screened_pairs_are_not_reported_as_incomplete(self):
        tree = ast.parse(Path(__file__).with_name("benchmark_guardrails.py").read_text())
        cell = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "case_failure_distribution")
        cell.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[cell], type_ignores=[]), "failure_report", "exec"), namespace)
        rendered = []
        mo = SimpleNamespace(md=lambda text: text, ui=SimpleNamespace(table=lambda rows: rows),
                             vstack=lambda items: rendered.extend(items))
        result = {"selected_guardrails": ["guardagent", "piguard", "adasteer"], "screened_case_ids": ["screened"],
                  "rows": [{"case_id": "screened", "expert": "guardagent", "succeeded": True},
                           *[{"case_id": "human", "expert": expert, "succeeded": False}
                             for expert in ("guardagent", "piguard", "adasteer")]]}
        namespace["case_failure_distribution"](result, mo)
        text = "\n".join(item for item in rendered if isinstance(item, str))
        self.assertIn("Cases with incomplete outcomes: 0", text)
        self.assertIn("Cases screened out (excluded from failure counts): 1", text)
        self.assertIn("Cases all selected guardrails failed on: 1", text)

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

    def test_screening_config_is_opt_in_generated_and_all_experts(self):
        generated = self.workspace / "generated.parquet"
        generated.write_bytes(b"generated")
        self.rows[generated.name] = [{"case_id": "gen:a", "prompt": "generated prompt"}]
        self.namespace["resolve_guardagent_policy"] = lambda source: (self.bundle, {"sha256": "guard"}, object())
        kwargs = dict(test_set="Aegis generated cases", selected_guardrails=["adasteer", "piguard", "guardagent"],
                      adasteer_path=self.bundle, piguard_path=self.bundle, guardagent_path=self.bundle,
                      output_root=self.workspace / "out", server_settings={"model": "judge", "port": 8000},
                      request_timeout=30, generated_path=generated)
        prepare = self.namespace["prepare_run"]
        normal = prepare(**kwargs)
        explicit = prepare(**kwargs, screening=False)
        screened = prepare(**kwargs, screening=True)
        self.assertEqual(normal["run_config"], explicit["run_config"])
        self.assertEqual(normal["fingerprint"], explicit["fingerprint"])
        self.assertNotIn("screening_order", normal["run_config"])
        self.assertEqual(screened["run_config"]["screening_order"], ["guardagent", "piguard", "adasteer"])
        self.assertNotEqual(screened["fingerprint"], normal["fingerprint"])
        changed = dict(screened["run_config"])
        changed.pop("screening_order")
        self.assertEqual(changed, normal["run_config"])
        for override in ({"selected_guardrails": ["piguard"]}, {"test_set": "Aegis unseen 10,000"}):
            with self.assertRaisesRegex(ValueError, "Screening requires"):
                prepare(**{**kwargs, **override}, screening=True)


class CascadeExecutionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calls = []
        self.fail_guard = False
        owner = self

        class Gate:
            def __init__(self, method):
                self.method = method

            def run(self, prompt):
                owner.calls.append((self.method, prompt))
                if owner.fail_guard and self.method == "guardagent" and prompt == "ga":
                    raise RuntimeError("transient gate error")
                blocked = prompt in {"ga", "pig", "human"} if self.method == "guardagent" else prompt == "human"
                return SimpleNamespace(block=blocked, metadata={})

            def close(self):
                pass

        class Runtime:
            laws = {"rd": None, "hd": None}

            def positions(self, prompt):
                return 0, 0

            def generate(self, prompt, rd, hd):
                owner.calls.append(("generate", prompt))
                return prompt + " response"

            def close(self):
                pass

        self.server = SimpleNamespace(process=object())
        def stop():
            self.calls.append(("stop", None))
            self.server.process = None
        self.server.stop = stop
        def ready(*args, **kwargs):
            self.calls.append(("judge_ready", None))
            self.server.process = object()
            return None, 0
        def judge(endpoint, model, prompt, response, timeout):
            self.calls.append(("judge", prompt))
            return {"attack_followed": False, "safe": True, "refusal": False, "useful": False}
        self.namespace = {
            "Path": Path, "json": json, "tempfile": tempfile, "os": os, "time": time,
            "REFUSAL_TEXT": "refusal", "RESPONDER_MODEL_ID": "responder", "error_text": str,
            "PIGuardArtifact": lambda *args, **kwargs: Gate("piguard"),
            "AdaSteer": lambda *args, **kwargs: SimpleNamespace(runtime=Runtime()),
            "predict_law": lambda *args: 0, "local_server_ready": ready, "judge_response": judge,
            "summarize_results": lambda rows: [],
        }
        tree = ast.parse(Path(__file__).with_name("benchmark_guardrails.py").read_text())
        cell = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "benchmark_helpers")
        names = {"run_screened_benchmark", "run_benchmark", "common_result_fields", "group_results_by_prompt",
                 "assumed_gate_judgment", "case_data", "load_latest", "append_event", "ensure_run_config", "choose_smoke_cases"}
        functions = [node for node in cell.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=functions, type_ignores=[]), "cascade_helpers", "exec"), self.namespace)
        cases = [{"case_id": name, "source_dataset": "generated", "source_index": index,
                  "prompt": name, "prompt_harm_label": "harmful" if name in {"ga", "ada"} else "unharmful",
                  "adversarial": None, "subcategory": "test"} for index, name in enumerate(("ga", "pig", "ada", "human"))]
        self.run = {
            "cases": cases, "dataset": "aegis", "fingerprint": "test", "selected_guardrails": ("adasteer", "piguard", "guardagent"),
            "run_config": {"screening_order": ["guardagent", "piguard", "adasteer"], "judge_model": "judge",
                           "local_endpoint": "endpoint", "responding_max_new_tokens": 128},
            "output_dir": self.root, "guardagent_policy": Gate("guardagent"), "piguard_path": self.root,
            "cache_root": self.root, "adasteer_source": self.root, "adasteer_bundle": self.root, "request_timeout": 30,
            **{key: self.root / name for key, name in (("results_path", "results.jsonl"), ("gate_path", "gates.jsonl"),
                                                       ("response_path", "responses.jsonl"), ("run_config_path", "run_config.json"))},
        }

    def execute(self, mode="full"):
        return self.namespace["run_benchmark"](self.run, mode=mode, server=self.server, server_settings={}, token=None)

    def test_stage_successes_are_final_gpu_order_and_resume(self):
        result = self.execute()
        self.assertTrue(result["complete"])
        self.assertEqual(result["expected_results"], 9)
        self.assertEqual(result["recorded_results"], 9)
        self.assertEqual(result["screened_case_ids"], ["ga", "pig"])
        self.assertEqual([stage["eligible"] for stage in result["screening_stages"]], [4, 3, 2])
        self.assertEqual([stage["screened_out"] for stage in result["screening_stages"]], [0, 1, 2])
        self.assertEqual([prompt for method, prompt in self.calls if method == "piguard"], ["pig", "ada", "human"])
        self.assertEqual([prompt for method, prompt in self.calls if method == "judge"], ["ada", "human"])
        self.assertLess(self.calls.index(("stop", None)), self.calls.index(("piguard", "pig")))
        self.assertLess(self.calls.index(("generate", "human")), self.calls.index(("judge_ready", None)))
        self.calls.clear()
        self.assertTrue(self.execute()["complete"])
        self.assertEqual(self.calls, [])

    def test_error_stops_cascade_then_only_failed_stage_work_resumes(self):
        self.fail_guard = True
        failed = self.execute()
        self.assertFalse(failed["complete"])
        self.assertEqual(len(failed["screening_stages"]), 1)
        self.assertFalse(any(method == "piguard" for method, _ in self.calls))
        self.fail_guard = False
        self.calls.clear()
        self.assertTrue(self.execute()["complete"])
        self.assertEqual([prompt for method, prompt in self.calls if method == "guardagent"], ["ga"])

    def test_smoke_filtered_stage_need_not_have_both_labels(self):
        result = self.execute(mode="smoke")
        self.assertTrue(result["complete"])
        self.assertEqual([stage["eligible"] for stage in result["screening_stages"]], [2, 1, 0])


if __name__ == "__main__":
    unittest.main()
