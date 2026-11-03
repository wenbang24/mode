"""Routing supervision and reporting checks; no model downloads or GPU required."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import laya_router as lr


def make_cases(count=20, prefix="train"):
    return [{"case_id": f"{prefix}-{index:04}", "source_dataset": "aegis", "source_index": index,
             "prompt": f"{prefix} prompt {index}", "prompt_harm_label": "harmful" if index % 2 else "unharmful",
             "adversarial": None, "subcategory": None} for index in range(count)]


def make_benchmark(directory, cases, parquet, *, split="unseen", version=4):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": version, "dataset": "aegis", "data_split": split,
        "test_sha256": lr.sha256_file(parquet), "case_count": len(cases),
        "selected_guardrails": list(lr.EXPERTS),
        **{key: str(index) * 64 for index, key in enumerate(lr.ARTIFACT_KEYS, 1)},
    }
    manifest["run_fingerprint"] = lr.fingerprint(manifest)
    lr.write_json(directory / "run_config.json", manifest)
    rows = []
    for index, case in enumerate(cases):
        for slot, expert in enumerate(lr.EXPERTS):
            rows.append({**case, "dataset": "aegis", "expert": expert,
                         "run_fingerprint": manifest["run_fingerprint"], "error": None,
                         "succeeded": slot == index % 3, "guardrail_seconds": float(slot + 1)})
    write_results(directory, rows)
    return rows


def write_results(directory, rows):
    (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


class RouterDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parquet = self.root / "original.parquet"
        self.parquet.write_bytes(b"test parquet content")
        self.cases = make_cases()
        self.run = self.root / "benchmark"
        self.rows = make_benchmark(self.run, self.cases, self.parquet)
        self.reader = patch.object(lr, "read_cases", return_value=self.cases)
        self.reader.start()
        self.addCleanup(self.reader.stop)

    def load(self, split="unseen"):
        return lr.load_benchmark(self.run, self.parquet, expected_rows=len(self.cases), split=split)

    def test_fastest_successful_and_ties_ignore_failed_experts(self):
        outcomes = {expert: {"succeeded": True, "guardrail_seconds": 3} for expert in lr.EXPERTS}
        self.assertEqual(lr.fastest_successful(outcomes), "adasteer")
        outcomes["adasteer"] = {"succeeded": False, "guardrail_seconds": 0}
        outcomes["guardagent"]["guardrail_seconds"] = 1
        self.assertEqual(lr.fastest_successful(outcomes), "guardagent")
        for outcome in outcomes.values():
            outcome["succeeded"] = False
        self.assertIsNone(lr.fastest_successful(outcomes))

    def test_judge_confidence_ranks_adasteer(self):
        outcomes = {
            "adasteer": {"succeeded": True, "guardrail_seconds": 1.3, "judge_confidence": 0.99},
            "piguard": {"succeeded": True, "guardrail_seconds": 0.03, "guard_blocked": True,
                        "guard_metadata": {"harmful_score": 0.9}},
            "guardagent": {"succeeded": False, "guardrail_seconds": 0.0},
        }
        self.assertEqual(lr.most_confident_successful(outcomes), "adasteer")
        outcomes["adasteer"]["judge_confidence"] = 0.5
        self.assertEqual(lr.most_confident_successful(outcomes), "piguard")
        outcomes["adasteer"]["judge_confidence"] = 1.5
        self.assertIsNone(lr.decision_confidence(outcomes["adasteer"]))

    def test_most_confident_successful_prefers_confident_gate_then_speed(self):
        outcomes = {
            "adasteer": {"succeeded": True, "guardrail_seconds": 1.3},
            "piguard": {"succeeded": True, "guardrail_seconds": 0.03, "guard_blocked": False,
                        "guard_metadata": {"harmful_score": 0.02}},
            "guardagent": {"succeeded": True, "guardrail_seconds": 0.0, "guard_blocked": False,
                           "guard_metadata": {"policy_score": 0.1}},
        }
        self.assertAlmostEqual(lr.decision_confidence(outcomes["piguard"]), 0.98)
        self.assertEqual(lr.most_confident_successful(outcomes), "piguard")
        outcomes["guardagent"]["guard_metadata"]["policy_score"] = 0.02
        self.assertEqual(lr.most_confident_successful(outcomes), "guardagent")
        outcomes["piguard"]["succeeded"] = outcomes["guardagent"]["succeeded"] = False
        self.assertEqual(lr.most_confident_successful(outcomes), "adasteer")
        outcomes["adasteer"]["succeeded"] = False
        self.assertIsNone(lr.most_confident_successful(outcomes))

    def test_manifest_target_rule_selects_supervision(self):
        manifest = json.loads((self.run / "run_config.json").read_text())
        manifest.pop("run_fingerprint")
        manifest["target_rule"] = "most_confident_successful"
        manifest["run_fingerprint"] = lr.fingerprint(manifest)
        lr.write_json(self.run / "run_config.json", manifest)
        for row in self.rows:
            row["run_fingerprint"] = manifest["run_fingerprint"]
            row["succeeded"] = True
            if row["expert"] != "adasteer":
                row["guard_blocked"] = False
                row["guard_metadata"] = {"harmful_score" if row["expert"] == "piguard" else "policy_score":
                                         0.1 if row["expert"] == "piguard" else 0.4}
        write_results(self.run, self.rows)
        self.assertEqual(set(self.load().targets.values()), {"piguard"})
        manifest["target_rule"] = "unknown"
        manifest["run_fingerprint"] = lr.fingerprint({k: v for k, v in manifest.items() if k != "run_fingerprint"})
        lr.write_json(self.run / "run_config.json", manifest)
        with self.assertRaisesRegex(ValueError, "target_rule"):
            self.load()

    def test_unicode_line_separators_remain_inside_jsonl_prompts(self):
        self.cases[0]["prompt"] = "Prompt with\u2028Unicode\u0085separators"
        for row in self.rows:
            if row["case_id"] == self.cases[0]["case_id"]:
                row["prompt"] = self.cases[0]["prompt"]
        (self.run / "results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows),
            encoding="utf-8",
        )
        self.assertEqual(self.load().cases[0]["prompt"], self.cases[0]["prompt"])

    def test_last_jsonl_result_wins_over_stale_grouped_file(self):
        revised = {**self.rows[0], "succeeded": False}
        write_results(self.run, [*self.rows, revised])
        lr.write_json(self.run / "prompt_results.json", [{"stale": True}])
        self.assertEqual(self.load().targets[self.cases[0]["case_id"]], lr.HUMAN)
        write_results(self.run, [*self.rows, {**revised, "error": "retry failed"}])
        with self.assertRaisesRegex(ValueError, "errored or unknown"):
            self.load()

    def test_grouped_results_restore_prompt(self):
        grouped = [{"prompt": case["prompt"], "results": [
            {key: value for key, value in row.items() if key != "prompt"}
            for row in self.rows if row["case_id"] == case["case_id"]]} for case in self.cases]
        lr.write_json(self.run / "prompt_results.json", grouped)
        (self.run / "results.jsonl").unlink()
        self.assertEqual(len(self.load().targets), len(self.cases))

    def test_rejects_missing_expert_errors_and_invalid_timings(self):
        write_results(self.run, self.rows[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.load()
        for field, value, message in (
            ("error", "failed", "errored or unknown"),
            ("succeeded", None, "errored or unknown"),
            ("guardrail_seconds", -1, "Invalid guardrail_seconds"),
            ("guardrail_seconds", float("nan"), "Invalid guardrail_seconds"),
            ("guardrail_seconds", float("inf"), "Invalid guardrail_seconds"),
            ("guardrail_seconds", True, "Invalid guardrail_seconds"),
        ):
            with self.subTest(field=field, value=value):
                rows = copy.deepcopy(self.rows)
                rows[0][field] = value
                write_results(self.run, rows)
                with self.assertRaisesRegex(ValueError, message):
                    self.load()

    def test_rejects_mixed_fingerprints_prompts_labels_and_datasets(self):
        for key, value in (("run_fingerprint", "another run"), ("prompt", "changed"),
                           ("prompt_harm_label", "changed"), ("dataset", "wildguard")):
            with self.subTest(key=key):
                rows = copy.deepcopy(self.rows)
                rows[0][key] = value
                write_results(self.run, rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_wrong_parquet_and_manifest_tampering(self):
        self.parquet.write_bytes(b"different original prompts")
        with self.assertRaisesRegex(ValueError, "parquet hash"):
            self.load()
        manifest = json.loads((self.run / "run_config.json").read_text())
        manifest["case_count"] = 5
        lr.write_json(self.run / "run_config.json", manifest)
        with self.assertRaisesRegex(ValueError, "fingerprint is invalid"):
            self.load()

    def test_test_only_legacy_schema_compatibility(self):
        make_benchmark(self.run, self.cases, self.parquet, split="test", version=3)
        self.assertEqual(len(self.load(split="test").cases), len(self.cases))
        with self.assertRaisesRegex(ValueError, "schema 4"):
            self.load()

    def test_preparation_preserves_all_failed_validation_and_only_prompt_inputs(self):
        self.rows[0]["succeeded"] = False
        write_results(self.run, self.rows)
        test_parquet = self.root / "test.parquet"
        test_parquet.write_bytes(b"held-out parquet")
        with patch.object(lr, "read_cases", side_effect=[self.cases, make_cases(prefix="test")]):
            prepared = lr.prepare_data(self.run, self.parquet, test_parquet, self.root / "artifacts")
        directory = Path(prepared["directory"])
        for split in ("train", "validation"):
            rows = [json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines()]
            self.assertTrue(all(set(row) == {"state", "questions", "expected"} for row in rows))
            self.assertTrue(all(isinstance(row["state"], str) for row in rows))
        splits = prepared["manifest"]["splits"]
        self.assertEqual(splits["train"]["cases"], 16)
        self.assertEqual(splits["validation"]["cases"], 4)
        self.assertEqual(sum(info["all_failed"] for info in splits.values()), 1)
        self.assertEqual(sum(info["labeled"] for info in splits.values()), 20)
        self.assertTrue((directory / "train_cases.json").is_file())


class SplitAndEvaluationTest(unittest.TestCase):
    def test_split_is_order_independent_stratified_and_disjoint(self):
        cases = make_cases(1000)
        splits = lr.split_cases(cases)
        self.assertEqual(splits, lr.split_cases(list(reversed(cases))))
        self.assertEqual(len(splits["train"]), 800)
        self.assertEqual(sum(row["prompt_harm_label"] == "harmful" for row in splits["train"]), 400)
        self.assertEqual(len(splits["validation"]), 200)
        lr.assert_disjoint(*splits.values())

    def test_rejects_normalized_prompt_and_id_leakage(self):
        left = make_cases(1)
        for right in ([{**left[0], "prompt": "different"}],
                      [{**left[0], "case_id": "new", "prompt": "  TRAIN   PROMPT 0  "}]):
            with self.assertRaisesRegex(ValueError, "overlap"):
                lr.assert_disjoint(left, right)

    def test_expert_artifact_and_behavior_mismatches_are_rejected(self):
        manifest = {key: "same" for key in lr.ARTIFACT_KEYS + lr.BEHAVIOR_KEYS}
        lr.check_matching_experts(manifest, manifest)
        for key in manifest:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                lr.check_matching_experts(manifest, {**manifest, key: "changed"})

    def test_success_denominator_includes_all_failed_prompts(self):
        cases = make_cases(2)
        outcomes = {case["case_id"]: {expert: {"succeeded": index == 0, "guardrail_seconds": slot + 1}
                    for slot, expert in enumerate(lr.EXPERTS)} for index, case in enumerate(cases)}
        targets = {case_id: lr.fastest_successful(values) or lr.HUMAN for case_id, values in outcomes.items()}
        predictions = [{"case_id": case["case_id"], "expert": "adasteer", "router_seconds": 0.1} for case in cases]
        report = lr.summarize_policy("trained", cases, outcomes, targets, predictions)[0]
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["all_failed"], 1)
        self.assertEqual(report["routing_accuracy"], 0.5)
        self.assertEqual(report["selected_expert_success"], 0.5)
        self.assertAlmostEqual(report["mean_estimated_cost_seconds"], 1.1)
        escalated = [predictions[0], {**predictions[1], "expert": lr.HUMAN}]
        report = lr.summarize_policy("trained", cases, outcomes, targets, escalated)[0]
        self.assertEqual((report["routing_accuracy"], report["selected_expert_success"]), (1, 1))
        self.assertEqual(report["human_escalations"], 1)
        self.assertAlmostEqual(report["mean_estimated_cost_seconds"], 0.6)
        wrongly_escalated = [{**predictions[0], "expert": lr.HUMAN}, predictions[1]]
        report = lr.summarize_policy("trained", cases, outcomes, targets, wrongly_escalated)[0]
        self.assertEqual(report["selected_expert_success"], 0)
        with self.assertRaisesRegex(ValueError, "every evaluation case"):
            lr.summarize_policy("trained", cases, outcomes, targets, predictions[:1])

    def test_router_reload_schema_probabilities_and_prompt_validation(self):
        answer = {"answers": {"expert": {"choice": "piguard", "probabilities": {
            "adasteer": 0.1, "piguard": 0.7, "guardagent": 0.2, "human": 0.0}}}, "usage": {"truncated": True}}
        agent = SimpleNamespace(predict=lambda *args, **kwargs: answer, __exit__=lambda *args: None)
        with tempfile.TemporaryDirectory() as temporary:
            lr.write_json(Path(temporary) / "questions.json", lr.QUESTIONS)
            with patch.dict("sys.modules", {"laya": SimpleNamespace(load=lambda *args, **kwargs: agent)}):
                with lr.MoDERouter(temporary, "cpu") as router:
                    result = router.predict("a prompt")
                    self.assertEqual(result["expert"], "piguard")
                    self.assertEqual(result["confidence"], 0.7)
                    self.assertTrue(result["truncated"])
                    for prompt in (None, "", "   "):
                        with self.assertRaises(ValueError):
                            router.predict(prompt)
                    answer["answers"]["expert"]["probabilities"]["piguard"] = float("nan")
                    with self.assertRaisesRegex(ValueError, "invalid choice probabilities"):
                        router.predict("a prompt")

    def test_export_copies_provenance_verifies_reload_and_prevents_overwrite(self):
        answer = {"answers": {"expert": {"choice": "piguard", "probabilities": {
            "adasteer": 0.1, "piguard": 0.7, "guardagent": 0.2, "human": 0.0}}}}
        agent = SimpleNamespace(predict=lambda *args, **kwargs: answer, __exit__=lambda *args: None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs/run-one"
            checkpoint = run / "checkpoint"
            checkpoint.mkdir(parents=True)
            lr.write_json(checkpoint / "questions.json", lr.QUESTIONS)
            (checkpoint / "model.safetensors").write_bytes(b"fixture model")
            (checkpoint / "checkpoint_latest").mkdir()
            (run / "training_data").mkdir()
            lr.write_json(run / "run.json", {"provenance": "fixture"})
            lr.write_json(run / "validation_report.json", {"fixture": True})
            with patch.dict("sys.modules", {"laya": SimpleNamespace(load=lambda *args, **kwargs: agent)}):
                exported = Path(lr.export_router(run, root, "cpu"))
                self.assertTrue((exported / "model.safetensors").is_file())
                self.assertFalse((exported / "checkpoint_latest").exists())
                self.assertEqual(json.loads((exported / "run.json").read_text()), {"provenance": "fixture"})
                self.assertEqual(json.loads((exported / "reload_check.json").read_text())["expert"], "piguard")
                with self.assertRaises(FileExistsError):
                    lr.export_router(run, root, "cpu")


if __name__ == "__main__":
    unittest.main()
