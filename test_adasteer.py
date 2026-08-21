import io
import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.benchmark_wildguard_train import prepare_cases, sample_cases
from scripts.experts.adasteer import AdaSteer, parse_compliance, parse_refusal
from scripts.experts.adasteer_bundle import (
    DIRECTION_GROUPS,
    DIRECTION_SIZE,
    EXPECTED_ROWS,
    GENERATION_BATCH_SIZE,
    GROUPS,
    HD_GRID,
    HD_MAX,
    HD_MIN,
    QWEN_HIDDEN_SIZE,
    QWEN_LAYERS,
    QwenSteeringRuntime,
    COMPLIANCE_PROMPT,
    REFUSAL_PROMPT,
    RD_GRID,
    RD_MAX,
    RD_MIN,
    TRAIN_ROWS,
    VALIDATION_ROWS,
    _calibrate_hd,
    _calibrate_rd,
    _evaluate,
    _install_bundle,
    _tune_hd,
    _tune_rd,
    behavior_group,
    finalize_directions,
    fit_law,
    predict_law,
    select_direction_rows,
    set_model_coefficients,
    sha256_file,
    stratified_split,
    validate_qwen_config,
    validate_rows,
    verify_bundle,
)


def rows_9000():
    return [
        {
            "source_index": index,
            "prompt": f"prompt-{index}",
            "prompt_harm_label": "harmful" if index % 3 else "unharmful",
            "adversarial": bool(index % 2),
            "subcategory": f"category-{index % 7}",
            "response_refusal_label": "refusal" if index % 5 else "compliance",
        }
        for index in range(EXPECTED_ROWS)
    ]


class FakeRuntime:
    def __init__(self):
        self.generated = 0

    def generate_batch(self, prompts, rd, hd, source_indices=None):
        self.generated += len(prompts)
        return [
            f"{source}|{rd_value:.8f}|{hd_value:.8f}"
            for source, rd_value, hd_value in zip(source_indices, rd, hd)
        ]


def positions(rows):
    return [
        {
            "source_index": row["source_index"],
            "rd_position": float(index),
            "hd_position": float(index),
        }
        for index, row in enumerate(rows)
    ]


class AdaSteerJudgeTest(unittest.TestCase):
    def test_exact_paper_parsers_reject_malformed_outputs(self):
        self.assertIs(parse_refusal("Yes"), True)
        self.assertIs(parse_refusal(" No\n"), False)
        self.assertIsNone(parse_refusal("yes"))
        self.assertIsNone(parse_refusal("Yes, it refuses"))
        for value in ("1_full_compliance", "2_full_refusal", "3_partial_refusal"):
            self.assertEqual(parse_compliance(value), value)
        self.assertIsNone(parse_compliance("CLASS: 1_full_compliance"))

    def test_paper_judge_uses_plain_exact_contract(self):
        def fake_urlopen(request, timeout):
            body = json.loads(request.data)
            self.assertEqual(body["temperature"], 0)
            self.assertIn("Reply with Yes or No.", body["messages"][0]["content"])
            return io.StringIO(
                json.dumps(
                    {
                        "model": "gpt-4o",
                        "choices": [{"message": {"content": "Yes"}}],
                    }
                )
            )

        context = SimpleNamespace(
            api_base="https://example.invalid/v1",
            api_key="unused",
            judge_model="openai/gpt-4o",
            seed=42,
        )
        with patch("scripts.experts.adasteer.urlopen", fake_urlopen):
            result = AdaSteer._paper_judge(
                context, "prompt", "response", "refusal", 1
            )
        self.assertIs(result["result"], True)


class AdaSteerDataTest(unittest.TestCase):
    def test_exact_deterministic_stratified_split(self):
        rows = rows_9000()
        train, validation = stratified_split(rows)
        train_again, validation_again = stratified_split(rows)
        self.assertEqual(len(train), TRAIN_ROWS)
        self.assertEqual(len(validation), VALIDATION_ROWS)
        self.assertEqual(
            [row["source_index"] for row in train],
            [row["source_index"] for row in train_again],
        )
        self.assertEqual(
            [row["source_index"] for row in validation],
            [row["source_index"] for row in validation_again],
        )
        self.assertFalse(
            {row["source_index"] for row in train}
            & {row["source_index"] for row in validation}
        )
        all_strata = Counter(
            (row["prompt_harm_label"], row["adversarial"], row["subcategory"])
            for row in rows
        )
        validation_strata = Counter(
            (row["prompt_harm_label"], row["adversarial"], row["subcategory"])
            for row in validation
        )
        self.assertTrue(
            all(
                abs(validation_strata[key] - count * 0.2) <= 1
                for key, count in all_strata.items()
            )
        )

    def test_stored_response_labels_cannot_change_qwen_grouping(self):
        first, second = rows_9000()[:2]
        first["prompt_harm_label"] = second["prompt_harm_label"] = "harmful"
        first["response_refusal_label"] = "refusal"
        second["response_refusal_label"] = "compliance"
        self.assertEqual(behavior_group(first, False), behavior_group(second, False))
        self.assertEqual(behavior_group(first, True), behavior_group(second, True))
        self.assertEqual(len(validate_rows(rows_9000())), EXPECTED_ROWS)

    def test_selects_13_per_required_qwen_group(self):
        grouped = {
            name: [
                {"source_index": offset * 100 + index, "prompt": str(index)}
                for index in range(20)
            ]
            for offset, name in enumerate(GROUPS)
        }
        selected = select_direction_rows(grouped)
        self.assertEqual(set(selected), set(DIRECTION_GROUPS))
        self.assertTrue(
            all(len(values) == DIRECTION_SIZE for values in selected.values())
        )
        self.assertEqual(selected, select_direction_rows(grouped))

    def test_benchmark_excludes_bundle_indices_and_stays_balanced(self):
        rows = [
            {
                "prompt": f"p-{index}",
                "prompt_harm_label": "harmful" if index % 2 else "unharmful",
            }
            for index in range(20)
        ]
        cases = prepare_cases(rows, {0, 1, 2, 3, 4, 5})
        sampled = sample_cases(cases, 10, 42)
        self.assertFalse({case.dataset_index for case in sampled} & set(range(6)))
        self.assertEqual(sum(case.malicious for case in sampled), 5)


class AdaSteerMathTest(unittest.TestCase):
    def test_context_overflow_reports_source_without_truncating(self):
        class Encoded(dict):
            def to(self, _device):
                return self

        class Tokenizer:
            kwargs = None

            def apply_chat_template(self, _messages, **_kwargs):
                return "formatted"

            def __call__(self, _prompts, **kwargs):
                self.kwargs = kwargs
                return Encoded(
                    input_ids=np.ones((1, 6), dtype=int),
                    attention_mask=np.ones((1, 6), dtype=int),
                )

        runtime = object.__new__(QwenSteeringRuntime)
        runtime.tokenizer = Tokenizer()
        runtime.max_new_tokens = 5
        runtime.context_window = 10
        with self.assertRaisesRegex(ValueError, "source 123 has 6 tokens"):
            runtime._encoded(["prompt"], [123])
        self.assertNotIn("truncation", runtime.tokenizer.kwargs)
        self.assertNotIn("max_length", runtime.tokenizer.kwargs)

    def test_qwen_directions_projection_and_fixed_law_bounds(self):
        shape = (QWEN_LAYERS, 2, QWEN_HIDDEN_SIZE)
        artifacts = finalize_directions(
            {
                "harmful_refusal": np.full(shape, 2.0, dtype=np.float32),
                "harmful_compliance": np.full(shape, 1.0, dtype=np.float32),
                "benign_compliance": np.full(shape, 3.0, dtype=np.float32),
            }
        )
        self.assertTrue(np.all(artifacts["RD/mean_diff.pkl"] == 1.0))
        self.assertTrue(np.all(artifacts["HD/mean_diff.pkl"] == 2.0))
        self.assertTrue(np.all(artifacts["HD/proj.pkl"] == 2.0))
        records = [
            {"position": float(index), "strength": 0.01 * index}
            for index in range(DIRECTION_SIZE)
        ]
        law = fit_law(records, "position", RD_MIN, RD_MAX)
        self.assertAlmostEqual(law["slope"], 0.01)
        self.assertEqual(predict_law(law, -100), RD_MIN)
        self.assertEqual(predict_law(law, 100), RD_MAX)

    def test_rd_and_hd_searches_use_paper_steps_and_bounds(self):
        calibration_rows = [
            {"source_index": index, "prompt": f"p-{index}"}
            for index in range(DIRECTION_SIZE)
        ]
        runtime = FakeRuntime()

        def refusal(_prompt, response):
            return float(response.split("|")[1]) >= 0.03

        def compliance(_prompt, response):
            return (
                "1_full_compliance"
                if float(response.split("|")[2]) <= -0.02
                else "2_full_refusal"
            )

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.jsonl"
            rd_law, rd_records = _calibrate_rd(
                runtime,
                calibration_rows,
                positions(calibration_rows),
                refusal,
                cache,
                "fingerprint",
                lambda _message: None,
            )
            self.assertEqual({record["strength"] for record in rd_records}, {0.03})
            self.assertEqual(
                (rd_law["minimum"], rd_law["maximum"]), (RD_MIN, RD_MAX)
            )
            hd_law, hd_records = _calibrate_hd(
                runtime,
                calibration_rows,
                positions(calibration_rows),
                rd_law,
                compliance,
                cache,
                "fingerprint",
                lambda _message: None,
            )
            self.assertEqual({record["strength"] for record in hd_records}, {-0.02})
            self.assertEqual(
                (hd_law["minimum"], hd_law["maximum"]), (HD_MIN, HD_MAX)
            )

    def test_grid_ties_are_deterministic_and_prefer_lower_steering(self):
        rows = [{"source_index": index, "prompt": str(index)} for index in range(2)]
        pos = [
            {
                "source_index": index,
                "rd_position": 0.0,
                "hd_position": 0.0,
            }
            for index in range(2)
        ]
        runtime = FakeRuntime()
        fitted_rd = {
            "slope": 0.0,
            "intercept": 0.1,
            "minimum": RD_MIN,
            "maximum": RD_MAX,
            "samples": 13,
        }
        fitted_hd = {
            "slope": 0.0,
            "intercept": 0.2,
            "minimum": HD_MIN,
            "maximum": HD_MAX,
            "samples": 13,
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "grid.jsonl"
            rd, rd_results, selected_rd = _tune_rd(
                runtime,
                rows,
                pos,
                fitted_rd,
                lambda _prompt, _response: True,
                cache,
                "fingerprint",
                lambda _message: None,
            )
            self.assertEqual(len(rd_results), len(RD_GRID))
            self.assertEqual(
                (rd["slope_multiplier"], rd["intercept_offset"]), (0.9, -0.01)
            )
            hd, hd_results, _ = _tune_hd(
                runtime,
                rows,
                pos,
                rows,
                pos,
                rd,
                fitted_hd,
                selected_rd["refusal_rate"],
                lambda _prompt, _response: True,
                lambda _prompt, _response: "1_full_compliance",
                cache,
                "fingerprint",
                lambda _message: None,
            )
            self.assertEqual(len(hd_results), len(HD_GRID))
            self.assertEqual(
                (hd["slope_multiplier"], hd["intercept_offset"]), (0.9, -0.05)
            )

    def test_batch_coefficients_follow_official_model_contract(self):
        captured = []

        class FakeTorch:
            float16 = "float16"

            @staticmethod
            def as_tensor(value, **kwargs):
                captured.append((value.tolist(), kwargs))
                return value.tolist()

        inner = SimpleNamespace(
            steer_vector=SimpleNamespace(device="cuda:0"),
            alpha_list=None,
            beta_list=None,
        )
        set_model_coefficients(
            SimpleNamespace(model=inner), FakeTorch, [-0.1, 0.2], [0.3, -0.4]
        )
        self.assertEqual(inner.alpha_list, [-0.1, 0.2])
        self.assertEqual(inner.beta_list, [0.3, -0.4])
        self.assertEqual(captured[0][1]["device"], "cuda:0")


class AdaSteerResumeAndBundleTest(unittest.TestCase):
    def test_evaluation_cache_resumes_and_rejects_stale_fingerprint(self):
        runtime = FakeRuntime()
        items = [
            {"source_index": index, "prompt": str(index), "rd": 0.1, "hd": 0.0}
            for index in range(GENERATION_BATCH_SIZE + 1)
        ]
        generator = lambda batch: runtime.generate_batch(
            [item["prompt"] for item in batch],
            [item["rd"] for item in batch],
            [item["hd"] for item in batch],
            [item["source_index"] for item in batch],
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.jsonl"
            first = _evaluate(
                items,
                generator,
                lambda _prompt, _response: True,
                "refusal",
                cache,
                "one",
                lambda _message: None,
            )
            generated = runtime.generated
            second = _evaluate(
                items,
                generator,
                lambda _prompt, _response: True,
                "refusal",
                cache,
                "one",
                lambda _message: None,
            )
            self.assertEqual(first, second)
            self.assertEqual(runtime.generated, generated)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                _evaluate(
                    items,
                    generator,
                    lambda _prompt, _response: True,
                    "refusal",
                    cache,
                    "two",
                    lambda _message: None,
                )

    def test_schema_v3_verification_and_schema_v2_rejection(self):
        shape = (QWEN_LAYERS, 2, QWEN_HIDDEN_SIZE)
        artifacts = finalize_directions(
            {
                "harmful_refusal": np.full(shape, 2.0),
                "harmful_compliance": np.full(shape, 1.0),
                "benign_compliance": np.full(shape, 3.0),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = {}
            for relative, value in artifacts.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as handle:
                    pickle.dump(value, handle)
                hashes[relative] = sha256_file(path)
            train = list(range(TRAIN_ROWS))
            validation = list(range(TRAIN_ROWS, EXPECTED_ROWS))
            law_rd = {
                "slope": 0.0,
                "intercept": 0.1,
                "minimum": RD_MIN,
                "maximum": RD_MAX,
                "samples": 13,
                "probe_layer": 5,
            }
            law_hd = {
                "slope": 0.0,
                "intercept": 0.0,
                "minimum": HD_MIN,
                "maximum": HD_MAX,
                "samples": 13,
                "probe_layer": 13,
            }
            metadata = {
                "schema_version": 3,
                "build_fingerprint": "f" * 64,
                "model_id": AdaSteer.model_id,
                "revision": None,
                "dataset_sha256": "0" * 64,
                "dataset_rows": EXPECTED_ROWS,
                "split_indices": {"train": train, "validation": validation},
                "qwen_group_counts": {
                    name: TRAIN_ROWS // 4 for name in GROUPS
                },
                "qwen_behavior_records": [
                    {
                        "source_index": index,
                        "group": GROUPS[index % 4],
                        "refusal": bool(index % 2),
                        "response_sha256": "0" * 64,
                    }
                    for index in train
                ],
                "selected_source_indices": {
                    name: list(range(offset * 13, offset * 13 + 13))
                    for offset, name in enumerate(DIRECTION_GROUPS)
                },
                "vector_shape": [QWEN_LAYERS, QWEN_HIDDEN_SIZE],
                "vector_dtype": "float16",
                "rd_probe_layer": 5,
                "hd_probe_layer": 13,
                "official_commit": "abc123",
                "dependencies": {
                    name: "test"
                    for name in ("numpy", "pyarrow", "torch", "transformers")
                },
                "coefficient_laws": {"rd": law_rd, "hd": law_hd},
                "calibration_records": {
                    name: [{"source_index": index} for index in range(13)]
                    for name in ("rd", "hd")
                },
                "validation": {
                    "rd_grid": [{}] * len(RD_GRID),
                    "hd_grid": [{}] * len(HD_GRID),
                },
                "generation": {
                    "system_prompt": "",
                    "do_sample": False,
                    "max_new_tokens": 128,
                    "batch_size": 32,
                },
                "judge": {
                    "model": "openai/gpt-4o",
                    "prompt_sha256": {
                        "refusal": hashlib.sha256(REFUSAL_PROMPT.encode()).hexdigest(),
                        "compliance": hashlib.sha256(COMPLIANCE_PROMPT.encode()).hexdigest(),
                    },
                },
                "artifact_sha256": hashes,
            }
            (root / "bundle.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(verify_bundle(root)["schema_version"], 3)
            metadata["schema_version"] = 2
            (root / "bundle.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rebuilt"):
                verify_bundle(root)

    def test_qwen_config_and_atomic_install(self):
        validate_qwen_config(
            {"model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 3584}
        )
        with self.assertRaises(ValueError):
            validate_qwen_config(
                {
                    "model_type": "qwen2",
                    "num_hidden_layers": 36,
                    "hidden_size": 4096,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, temporary = root / "bundle", root / "temporary"
            target.mkdir()
            temporary.mkdir()
            (target / "marker").write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _install_bundle(temporary, target, overwrite=False)
            self.assertEqual(
                (target / "marker").read_text(encoding="utf-8"), "old"
            )


if __name__ == "__main__":
    unittest.main()
