import io
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.experts.adasteer import AdaSteer
from scripts.experts.adasteer_bundle import (
    ARTIFACT_SHAPES,
    CALIBRATION_SIZE,
    EXPECTED_ROWS,
    GROUPS,
    QWEN_HIDDEN_SIZE,
    QWEN_LAYERS,
    _install_bundle,
    finalize_directions,
    fit_law,
    predict_law,
    set_model_coefficients,
    sha256_file,
    validate_qwen_config,
    validate_rows,
    verify_bundle,
)


class AdaSteerJudgeTest(unittest.TestCase):
    def test_reserves_enough_tokens_for_reasoning_and_json(self):
        def fake_urlopen(request, timeout):
            body = json.loads(request.data)
            enough_tokens = body["max_completion_tokens"] >= 1024
            content = (
                json.dumps(
                    {
                        "useful": True,
                        "safe": True,
                        "attack_followed": False,
                        "refusal": False,
                    }
                )
                if enough_tokens
                else None
            )
            return io.StringIO(
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop" if enough_tokens else "length",
                                "message": {"content": content},
                            }
                        ]
                    }
                )
            )

        context = SimpleNamespace(
            api_base="https://example.invalid/v1",
            api_key="unused",
            judge_model="fake-judge",
            seed=42,
        )
        with patch("scripts.experts.adasteer.urlopen", fake_urlopen):
            judgment = AdaSteer._judge(
                context, "prompt", "response", max_retries=1
            )

        self.assertTrue(judgment["safe"])
        self.assertEqual(judgment["attempts"], 1)


class AdaSteerBundleTest(unittest.TestCase):
    def test_validates_all_rows_without_deduplicating_prompts(self):
        rows = []
        labels = (
            ("harmful", "refusal"),
            ("harmful", "compliance"),
            ("unharmful", "compliance"),
            ("unharmful", "refusal"),
        )
        for index in range(EXPECTED_ROWS):
            prompt_label, action_label = labels[index % len(labels)]
            rows.append(
                {
                    "source_index": index,
                    "prompt": f"prompt-{index % 10}",
                    "response": "response",
                    "prompt_harm_label": prompt_label,
                    "response_harm_label": "unharmful",
                    "response_refusal_label": action_label,
                }
            )

        validated, counts = validate_rows(rows)

        self.assertEqual(len(validated), EXPECTED_ROWS)
        self.assertEqual(sum(counts.values()), EXPECTED_ROWS)
        self.assertEqual(set(counts), set(GROUPS))

    def test_builds_qwen_directions_and_projection(self):
        shape = (QWEN_LAYERS, 2, QWEN_HIDDEN_SIZE)
        activations = {
            "harmful_refusal": np.full(shape, 2.0, dtype=np.float32),
            "harmful_compliance": np.full(shape, 1.0, dtype=np.float32),
            "benign_compliance": np.full(shape, 3.0, dtype=np.float32),
            "benign_refusal": np.zeros(shape, dtype=np.float32),
        }

        artifacts = finalize_directions(activations)

        self.assertEqual(set(artifacts), set(ARTIFACT_SHAPES))
        self.assertTrue(np.all(artifacts["RD/mean_diff.pkl"] == 1.0))
        self.assertTrue(np.all(artifacts["HD/mean_diff.pkl"] == 2.0))
        self.assertTrue(np.all(artifacts["HD/proj.pkl"] == 2.0))
        self.assertTrue(np.all(artifacts["HD/class_a.pkl"] == 3.0))
        self.assertTrue(np.all(artifacts["HD/class_b.pkl"] == 1.0))
        self.assertTrue(all(value.dtype == np.float16 for value in artifacts.values()))

    def test_fits_and_clamps_coefficient_law(self):
        records = [
            {"position": float(index), "strength": 0.1 * index - 0.2}
            for index in range(CALIBRATION_SIZE)
        ]

        law = fit_law(records, "position")

        self.assertAlmostEqual(law["slope"], 0.1)
        self.assertAlmostEqual(law["intercept"], -0.2)
        self.assertEqual(predict_law(law, -100), law["minimum"])
        self.assertEqual(predict_law(law, 100), law["maximum"])

    def test_rejects_non_qwen_7b_config(self):
        validate_qwen_config(
            {"model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 3584}
        )
        with self.assertRaises(ValueError):
            validate_qwen_config(
                {"model_type": "qwen2", "num_hidden_layers": 36, "hidden_size": 4096}
            )

    def test_verifies_schema_two_bundle_and_hashes(self):
        shape = (QWEN_LAYERS, 2, QWEN_HIDDEN_SIZE)
        artifacts = finalize_directions(
            {
                "harmful_refusal": np.full(shape, 2.0),
                "harmful_compliance": np.full(shape, 1.0),
                "benign_compliance": np.full(shape, 3.0),
                "benign_refusal": np.zeros(shape),
            }
        )
        law = {
            "slope": 0.1,
            "intercept": 0.0,
            "minimum": -0.2,
            "maximum": 0.2,
            "samples": CALIBRATION_SIZE,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = {}
            for relative, value in artifacts.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as handle:
                    pickle.dump(value, handle)
                hashes[relative] = sha256_file(path)
            (root / "bundle.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "model_id": AdaSteer.model_id,
                        "revision": None,
                        "dataset_sha256": "0" * 64,
                        "dataset_rows": EXPECTED_ROWS,
                        "group_counts": {name: EXPECTED_ROWS // 4 for name in GROUPS[:-1]}
                        | {GROUPS[-1]: EXPECTED_ROWS - 3 * (EXPECTED_ROWS // 4)},
                        "vector_shape": [QWEN_LAYERS, QWEN_HIDDEN_SIZE],
                        "vector_dtype": "float16",
                        "rd_probe_layer": 5,
                        "hd_probe_layer": 13,
                        "official_commit": "abc123",
                        "dependencies": {
                            "numpy": "test",
                            "pyarrow": "test",
                            "torch": "test",
                            "transformers": "test",
                        },
                        "coefficient_laws": {
                            "rd": law | {"probe_layer": 5},
                            "hd": law | {"probe_layer": 13},
                        },
                        "calibration_records": {
                            name: [
                                {"source_index": index}
                                for index in range(CALIBRATION_SIZE)
                            ]
                            for name in ("rd", "hd")
                        },
                        "artifact_sha256": hashes,
                    }
                ),
                encoding="utf-8",
            )

            metadata = verify_bundle(root)

        self.assertEqual(metadata["schema_version"], 2)

    def test_sets_coefficients_on_official_model_contract(self):
        captured = []

        class FakeTorch:
            float16 = "float16"

            @staticmethod
            def tensor(value, **kwargs):
                captured.append((value, kwargs))
                return value

        inner = SimpleNamespace(
            steer_vector=SimpleNamespace(device="cuda:0"),
            alpha_list=None,
            beta_list=None,
        )
        model = SimpleNamespace(model=inner)

        set_model_coefficients(model, FakeTorch, -0.1, 0.2)

        self.assertEqual(inner.alpha_list, [-0.1])
        self.assertEqual(inner.beta_list, [0.2])
        self.assertEqual(captured[0][1]["device"], "cuda:0")

    def test_atomic_install_preserves_existing_bundle_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "bundle"
            temporary = root / "temporary"
            target.mkdir()
            temporary.mkdir()
            (target / "marker").write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _install_bundle(temporary, target, overwrite=False)
            self.assertEqual((target / "marker").read_text(encoding="utf-8"), "old")

    def test_atomic_install_restores_existing_bundle_after_replace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "bundle"
            temporary = root / "temporary"
            target.mkdir()
            temporary.mkdir()
            (target / "marker").write_text("old", encoding="utf-8")
            real_replace = os.replace
            calls = 0

            def fail_second_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated install failure")
                return real_replace(source, destination)

            with patch(
                "scripts.experts.adasteer_bundle.os.replace",
                side_effect=fail_second_replace,
            ):
                with self.assertRaises(OSError):
                    _install_bundle(temporary, target, overwrite=True)
            self.assertEqual((target / "marker").read_text(encoding="utf-8"), "old")


if __name__ == "__main__":
    unittest.main()
