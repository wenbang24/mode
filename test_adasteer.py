import io
import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.benchmark_wildguard_train import (
    bundle_source_indices,
    prepare_cases,
    sample_cases,
)
from scripts.experts.adasteer import AdaSteer, parse_compliance, parse_refusal
from scripts.experts.adasteer_bundle import (
    BUNDLE_SCHEMA_VERSION,
    DEFAULT_MODEL_ID,
    DIRECTION_GROUPS,
    DIRECTION_SIZE,
    EXPECTED_ROWS,
    GENERATION_BATCH_SIZE,
    GROUPS,
    HD_GRID,
    HD_MAX,
    HD_MIN,
    COMPLIANCE_PROMPT,
    REFUSAL_PROMPT,
    REQUIRED_CHECKOUT_FILES,
    RD_GRID,
    RD_MAX,
    RD_MIN,
    RUNTIME_PROVENANCE,
    SteeringRuntime,
    TRAIN_ROWS,
    _calibrate_hd,
    _calibrate_rd,
    _evaluate,
    _format_prompts,
    _install_bundle,
    _tune_hd,
    _tune_rd,
    artifact_shapes,
    behavior_group,
    decoder_layers,
    finalize_directions,
    fit_law,
    predict_law,
    select_direction_rows,
    sha256_file,
    summarize_test_metrics,
    validate_dataset_splits,
    validate_model_config,
    validate_rows,
    verify_bundle,
)


TEST_LAYERS = 4
TEST_WIDTH = 6
TEST_RD_PROBE_LAYER = 1
TEST_HD_PROBE_LAYER = 3


def rows_10000():
    return [
        {
            "case_id": f"test:{index}",
            "source_dataset": "test",
            "source_index": index,
            "prompt": f"prompt-{index}",
            "prompt_harm_label": "harmful" if index % 3 else "unharmful",
            "adversarial": bool(index % 2),
            "subcategory": f"category-{index % 7}",
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
    def test_adasteer_package_does_not_import_sibling_experts(self):
        import scripts.experts as experts_package

        self.assertIs(experts_package.AdaSteer, AdaSteer)
        self.assertNotIn("scripts.experts.guardagent", sys.modules)
        self.assertNotIn("scripts.experts.piguard_finetuned", sys.modules)
        self.assertNotIn("scripts.experts.piguard_guardrail", sys.modules)

    def test_paper_parsers_reject_malformed_outputs(self):
        for value in ("Yes", "Yes.", "yes", " YES! It refuses.", "Yes, it refuses"):
            with self.subTest(value=value):
                self.assertIs(parse_refusal(value), True)
        for value in (
            " No\n", "no", "No: it answers.",
            "No. The answer does not explicitly reject the user’s request. "
            "It attempts to provide a rewritten email, although it does not fully follow all instructions.",
        ):
            with self.subTest(value=value):
                self.assertIs(parse_refusal(value), False)
        for value in (None, 0, "", "Nobody", "Yesterday", "Verdict: No", "Yes/No", "No/Yes", "Yes or No", "No idea", "No.example"):
            with self.subTest(value=value):
                self.assertIsNone(parse_refusal(value))
        for value in ("1_full_compliance", "2_full_refusal", "3_partial_refusal"):
            self.assertEqual(parse_compliance(value), value)
            self.assertEqual(parse_compliance(value + "."), value)
        self.assertIsNone(parse_compliance("CLASS: 1_full_compliance"))

    def test_paper_judge_uses_plain_exact_contract(self):
        def fake_urlopen(request, timeout):
            body = json.loads(request.data)
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["max_completion_tokens"], 256)
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

    def test_paper_judge_retry_budget(self):
        context = SimpleNamespace(
            api_base="https://example.invalid/v1", api_key="unused",
            judge_model="gpt-4o", seed=42,
        )
        def reply(content):
            return io.StringIO(json.dumps({"choices": [{"message": {"content": content}}]}))

        def http_error(code):
            error = HTTPError(context.api_base, code, "error", {}, io.BytesIO())
            self.addCleanup(error.close)
            return error

        failures = [
            lambda: reply("unclear"), lambda: reply(None),
            lambda: io.StringIO("not JSON"), lambda: io.StringIO('{}'),
            lambda: io.StringIO('{"choices": []}'),
            lambda: URLError("offline"), lambda: TimeoutError(),
            *[lambda code=code: http_error(code)
              for code in (408, 409, 429, 500, 503)],
        ]
        for failure in failures:
            for contract, valid, expected in (("refusal", "No. It answers.", False), ("compliance", "1_full_compliance", "1_full_compliance")):
                with self.subTest(failure=failure, contract=contract), patch(
                    "scripts.experts.adasteer.urlopen", side_effect=[failure(), reply(valid)]
                ) as request, patch("scripts.experts.adasteer.time.sleep") as sleep:
                    result = AdaSteer._paper_judge(context, "prompt", "response", contract)
                    self.assertEqual(result["result"], expected)
                    self.assertEqual(result["attempts"], 2)
                    self.assertEqual(request.call_count, 2)
                    sleep.assert_called_once()

        for code in (400, 401, 403):
            with self.subTest(code=code), patch(
                "scripts.experts.adasteer.urlopen",
                side_effect=http_error(code),
            ) as request, patch("scripts.experts.adasteer.time.sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, f"after 1 attempts: HTTP {code}"):
                    AdaSteer._paper_judge(context, "prompt", "response", "refusal")
                self.assertEqual(request.call_count, 1)
                sleep.assert_not_called()

        # Exercise the trainer path too: exhausting the API budget must not restart it.
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.experts.adasteer.urlopen", side_effect=lambda *a, **kw: reply("unclear")
        ) as request, patch("scripts.experts.adasteer.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 5 attempts:.*malformed refusal"):
                _evaluate(
                    [{"source_index": 0, "prompt": "prompt", "rd": 0.0, "hd": 0.0}],
                    lambda batch: ["response"],
                    lambda prompt, response: AdaSteer._paper_judge(context, prompt, response, "refusal")["result"],
                    "refusal", Path(directory) / "cache.jsonl", "fingerprint",
                )
            self.assertEqual(request.call_count, 5)
            self.assertEqual(sleep.call_count, 4)
            self.assertFalse((Path(directory) / "cache.jsonl").exists())


class AdaSteerDataTest(unittest.TestCase):
    def test_dedicated_splits_use_all_training_rows_and_reject_leakage(self):
        def held_out(name, offset):
            return [
                {
                    "case_id": f"{name}:{index}",
                    "source_dataset": name,
                    "source_index": index,
                    "prompt": f"prompt-{offset + index}",
                    "prompt_harm_label": "harmful" if index % 2 else "unharmful",
                    "adversarial": None,
                    "subcategory": None,
                }
                for index in range(4)
            ]

        train = rows_10000()
        validation = held_out("validation", 10_000)
        test = held_out("test", 20_000)
        splits = validate_dataset_splits(train, validation, test)
        self.assertEqual(len(splits["train"]), TRAIN_ROWS)
        self.assertEqual(
            splits["validation"][0]["source_index"],
            splits["test"][0]["source_index"],
        )
        self.assertNotEqual(
            splits["validation"][0]["_cache_id"],
            splits["test"][0]["_cache_id"],
        )
        validation[0]["prompt"] = "  PROMPT-0  "
        with self.assertRaisesRegex(ValueError, "overlapping normalized prompts"):
            validate_dataset_splits(train, validation, test)

    def test_stored_response_labels_cannot_change_model_grouping(self):
        first, second = rows_10000()[:2]
        first["prompt_harm_label"] = second["prompt_harm_label"] = "harmful"
        first["response_refusal_label"] = "refusal"
        second["response_refusal_label"] = "compliance"
        self.assertEqual(behavior_group(first, False), behavior_group(second, False))
        self.assertEqual(behavior_group(first, True), behavior_group(second, True))
        self.assertEqual(len(validate_rows(rows_10000())), EXPECTED_ROWS)

    def test_nullable_source_metadata_is_valid(self):
        rows = rows_10000()
        for row in rows:
            row["adversarial"] = None
            row["subcategory"] = None
        self.assertEqual(len(validate_rows(rows)), TRAIN_ROWS)

    def test_held_out_metrics_cover_both_labels(self):
        metrics = summarize_test_metrics(
            [{"result": True}, {"result": False}],
            [
                {"result": "1_full_compliance"},
                {"result": "2_full_refusal"},
                {"result": "1_full_compliance"},
            ],
        )
        self.assertEqual(metrics["rows"], 5)
        self.assertEqual(metrics["harmful_refusal_rate"], 0.5)
        self.assertEqual(metrics["benign_full_compliance_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["overall_success_rate"], 0.6)

    def test_selects_13_per_required_behavior_group(self):
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
        metadata = {
            "datasets": {
                "train": {
                    "source_datasets": ["allenai/wildguardmix/wildguardtrain"],
                    "source_indices": [1, 2],
                },
                "validation": {
                    "source_datasets": ["allenai/wildguardmix/wildguardtrain"],
                    "source_indices": [3],
                },
                "test": {
                    "source_datasets": ["allenai/wildguardmix/wildguardtest"],
                    "source_indices": [1],
                },
            }
        }
        self.assertEqual(
            bundle_source_indices(
                metadata, "allenai/wildguardmix/wildguardtrain"
            ),
            {1, 2, 3},
        )


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

        runtime = object.__new__(SteeringRuntime)
        runtime.tokenizer = Tokenizer()
        runtime.max_new_tokens = 5
        runtime.context_window = 10
        with self.assertRaisesRegex(ValueError, "source 123 has 6 tokens"):
            runtime._encoded(["prompt"], [123])
        self.assertNotIn("truncation", runtime.tokenizer.kwargs)
        self.assertNotIn("max_length", runtime.tokenizer.kwargs)
        self.assertEqual(
            _format_prompts(SimpleNamespace(chat_template=None), ["plain"]),
            ["plain"],
        )

    def test_dynamic_directions_projection_and_fixed_law_bounds(self):
        shape = (TEST_LAYERS, 2, TEST_WIDTH)
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

    def test_decoder_hooks_apply_prefill_batches_skip_decode_and_clean_up(self):
        class Handle:
            def __init__(self, block):
                self.block = block

            def remove(self):
                self.block.hook = None

        class Block:
            hook = None

            def register_forward_hook(self, hook):
                self.hook = hook
                return Handle(self)

            def __call__(self, hidden):
                output = (hidden, "cache")
                return self.hook(self, (), output) if self.hook else output

        runtime = object.__new__(SteeringRuntime)
        runtime._rd_vectors = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        runtime._hd_vectors = np.array(
            [[2.0, 0.0, -2.0], [1.0, 1.0, 1.0]], dtype=np.float32
        )
        runtime._rd_coefficients = np.array([0.5, -1.0], dtype=np.float32)
        runtime._hd_coefficients = np.array([0.25, 2.0], dtype=np.float32)
        blocks = [Block(), Block()]
        runtime._attach_hooks(blocks)

        hidden = np.zeros((2, 2, 3), dtype=np.float32)
        steered, cache = blocks[0](hidden)
        expected = np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[3.0, -2.0, -7.0], [3.0, -2.0, -7.0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(steered, expected)
        self.assertEqual(cache, "cache")
        np.testing.assert_allclose(
            blocks[1](hidden)[0],
            np.array(
                [
                    [[2.25, 2.75, 3.25], [2.25, 2.75, 3.25]],
                    [[-2.0, -3.0, -4.0], [-2.0, -3.0, -4.0]],
                ],
                dtype=np.float32,
            ),
        )
        decode = np.zeros((2, 1, 3), dtype=np.float32)
        self.assertIs(blocks[0](decode)[0], decode)

        runtime._remove_hooks()
        self.assertTrue(all(block.hook is None for block in blocks))
        self.assertEqual(runtime._hook_handles, [])


class AdaSteerResumeAndBundleTest(unittest.TestCase):
    def test_cache_keys_isolate_split_local_source_indices(self):
        runtime = FakeRuntime()
        items = [
            {
                "cache_id": f"{split}:same-case",
                "source_index": 0,
                "prompt": split,
                "rd": 0.1,
                "hd": 0.0,
            }
            for split in ("validation", "test")
        ]
        generator = lambda batch: runtime.generate_batch(
            [item["prompt"] for item in batch],
            [item["rd"] for item in batch],
            [item["hd"] for item in batch],
            [item["source_index"] for item in batch],
        )
        with tempfile.TemporaryDirectory() as directory:
            records = _evaluate(
                items,
                generator,
                lambda _prompt, _response: True,
                "refusal",
                Path(directory) / "cache.jsonl",
                "fingerprint",
                lambda _message: None,
            )
        self.assertEqual(runtime.generated, 2)
        self.assertEqual(
            {record["cache_id"] for record in records},
            {"validation:same-case", "test:same-case"},
        )

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

    def test_schema_v5_verification_and_old_or_mismatched_rejection(self):
        shape = (TEST_LAYERS, 2, TEST_WIDTH)
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
            law_rd = {
                "slope": 0.0,
                "intercept": 0.1,
                "minimum": RD_MIN,
                "maximum": RD_MAX,
                "samples": 13,
                "probe_layer": TEST_RD_PROBE_LAYER,
            }
            law_hd = {
                "slope": 0.0,
                "intercept": 0.0,
                "minimum": HD_MIN,
                "maximum": HD_MAX,
                "samples": 13,
                "probe_layer": TEST_HD_PROBE_LAYER,
            }
            metadata = {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "build_fingerprint": "f" * 64,
                "model_id": AdaSteer.model_id,
                "revision": None,
                "model_type": "llama",
                "num_hidden_layers": TEST_LAYERS,
                "hidden_size": TEST_WIDTH,
                "steering_runtime": RUNTIME_PROVENANCE,
                "datasets": {
                    "train": {
                        "path": "train.parquet",
                        "rows": TRAIN_ROWS,
                        "sha256": "0" * 64,
                        "source_datasets": ["source/train"],
                        "source_indices": train,
                    },
                    "validation": {
                        "path": "validation.parquet",
                        "rows": 4,
                        "sha256": "1" * 64,
                        "source_datasets": ["source/validation"],
                        "source_indices": list(range(4)),
                    },
                    "test": {
                        "path": "test.parquet",
                        "rows": 4,
                        "sha256": "2" * 64,
                        "source_datasets": ["source/test"],
                        "source_indices": list(range(4)),
                    },
                },
                "behavior_group_counts": {
                    name: TRAIN_ROWS // 4 for name in GROUPS
                },
                "behavior_records": [
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
                "vector_shape": [TEST_LAYERS, TEST_WIDTH],
                "vector_dtype": "float16",
                "rd_probe_layer": TEST_RD_PROBE_LAYER,
                "hd_probe_layer": TEST_HD_PROBE_LAYER,
                "official_commit": "abc123",
                "official_source_sha256": {
                    name: "a" * 64 for name in REQUIRED_CHECKOUT_FILES
                },
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
                "test_metrics": {
                    "rows": 4,
                    "harmful_rows": 2,
                    "benign_rows": 2,
                    "harmful_refusal_rate": 0.5,
                    "benign_full_compliance_rate": 1.0,
                    "overall_success_rate": 0.75,
                    "balanced_success_rate": 0.75,
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
            self.assertEqual(
                verify_bundle(root)["schema_version"], BUNDLE_SCHEMA_VERSION
            )
            metadata["schema_version"] = 4
            (root / "bundle.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rebuilt"):
                verify_bundle(root)
            metadata["schema_version"] = BUNDLE_SCHEMA_VERSION
            metadata["hidden_size"] += 1
            (root / "bundle.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "vector contract"):
                verify_bundle(root)

    def test_decoder_model_validation_and_atomic_install(self):
        self.assertEqual(DEFAULT_MODEL_ID, "Qwen/Qwen2.5-3B-Instruct")
        qwen_geometry = validate_model_config(
            {"model_type": "qwen2", "num_hidden_layers": 36, "hidden_size": 2048}
        )
        self.assertEqual(qwen_geometry["num_hidden_layers"], 36)
        self.assertEqual(
            validate_model_config(
                {"model_type": "llama", "num_hidden_layers": 3, "hidden_size": 7},
                0,
                2,
            )["hidden_size"],
            7,
        )
        for config, rd_layer, hd_layer in (
            (
                {
                    "model_type": "t5",
                    "is_encoder_decoder": True,
                    "num_hidden_layers": 12,
                    "hidden_size": 768,
                },
                1,
                2,
            ),
            ({"model_type": "llama", "num_hidden_layers": 0, "hidden_size": 8}, 0, 0),
            ({"model_type": "llama", "num_hidden_layers": 2, "hidden_size": -1}, 0, 1),
            ({"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 8}, 0, 2),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    validate_model_config(config, rd_layer, hd_layer)

        class Block:
            def register_forward_hook(self, _hook):
                return None

        layers = [Block(), Block()]
        model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        self.assertIs(decoder_layers(model, 2), layers)
        with self.assertRaisesRegex(ValueError, r"\.model\.layers"):
            decoder_layers(SimpleNamespace(), 2)
        with self.assertRaisesRegex(ValueError, "expected 3"):
            decoder_layers(model, 3)
        self.assertEqual(
            artifact_shapes(2, 7)["HD/class_b.pkl"], (2, 1, 7)
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
