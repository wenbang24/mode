"""CPU-only GuardAgent adapter and policy-workflow checks."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.experts.base import ExpertOutcome
from scripts.experts.guardagent import (
    ALLOW_CODE,
    BLOCK_CODE,
    MEMORY,
    MEMORY_SHOTS,
    FrozenGuardAgent,
    GuardAgent,
    GuardAgentTrainer,
    compile_frozen_policies,
    load_policy_memory,
    restricted_execute,
)
from scripts.experts import guardagent_policy as policy
from scripts.training_cases import write_cases


def case(case_id, prompt, label, source_index):
    return {
        "case_id": case_id,
        "source_dataset": "fixture",
        "source_index": source_index,
        "prompt": prompt,
        "prompt_harm_label": label,
        "adversarial": label == "harmful",
        "subcategory": "unsafe" if label == "harmful" else "benign",
    }


class RestrictedExecutorTest(unittest.TestCase):
    def test_valid_allow_and_block(self):
        self.assertFalse(restricted_execute(ALLOW_CODE, "hello")["access_denied"])
        self.assertTrue(restricted_execute(BLOCK_CODE, "bad")["access_denied"])

    def test_rejects_unsafe_syntax_and_invalid_results(self):
        invalid = (
            "import os\n" + ALLOW_CODE,
            ALLOW_CODE.replace("AGENT_OUTPUT", "open('x')", 1),
            ALLOW_CODE.replace("AGENT_OUTPUT", "answer.upper()", 1),
            ALLOW_CODE.replace("AGENT_OUTPUT", "unknown", 1),
            "if True:\n    access_denied = True\n" + BLOCK_CODE,
            ALLOW_CODE.replace("candidate_text = AGENT_OUTPUT\n", ""),
            BLOCK_CODE.replace("guardrailed_answer = None", "guardrailed_answer = answer"),
            ALLOW_CODE.replace("guardrailed_answer = answer", "guardrailed_answer = None"),
        )
        for code in invalid:
            with self.subTest(code=code), self.assertRaises((SyntaxError, ValueError)):
                restricted_execute(code, "prompt")


class AdapterTest(unittest.TestCase):
    def test_frozen_rules_classify_unseen_prompts_without_model_calls(self):
        self.assertIs(GuardAgent, FrozenGuardAgent)
        memory = [
            *MEMORY,
            {"agent input": "review", "agent output": "bad train request", "subtasks": "block", "code": BLOCK_CODE},
            {"agent input": "review", "agent output": "good train request", "subtasks": "allow", "code": ALLOW_CODE},
        ]
        guard = FrozenGuardAgent(policies=compile_frozen_policies(memory))
        self.assertTrue(guard.run("bad unseen request").block)
        self.assertFalse(guard.run("good unseen request").block)

    def test_correct_only_learning(self):
        agent = object.__new__(GuardAgentTrainer)
        agent.memory = [dict(item) for item in MEMORY]
        agent.num_shots = MEMORY_SHOTS
        updates = []
        agent.guard = SimpleNamespace(update_memory=lambda shots, memory: updates.append((shots, list(memory))))
        good = ExpertOutcome(False, metadata={"task_plan": "safe plan", "generated_code": ALLOW_CODE})
        bad = ExpertOutcome(True, metadata={"task_plan": "wrong plan", "generated_code": BLOCK_CODE})
        with patch.object(GuardAgentTrainer, "run", side_effect=[good, bad]):
            outcome, learned = agent.learn("hello", False)
            self.assertFalse(outcome.block)
            self.assertEqual(learned["subtasks"], "safe plan")
            self.assertEqual(len(agent.memory), 3)
            outcome, learned = agent.learn("benign", False)
            self.assertTrue(outcome.block)
            self.assertIsNone(learned)
            self.assertEqual(len(agent.memory), 3)
        self.assertEqual(updates[-1][0], MEMORY_SHOTS)

    def test_nearest_examples_are_stable_and_three_shot(self):
        agent = object.__new__(GuardAgentTrainer)
        agent.official = SimpleNamespace(
            Levenshtein=SimpleNamespace(distance=lambda left, right: abs(len(left) - len(right)))
        )
        agent._retrieved_indexes = []
        memory = [
            {"agent input": "x", "agent output": value, "subtasks": f"plan-{index}", "code": ALLOW_CODE}
            for index, value in enumerate(("a", "b", "ccc", "dddd"))
        ]
        official = SimpleNamespace(memory=memory, num_shots=3)
        text = agent._retrieve_examples(official, "x", "z")
        self.assertEqual(agent._retrieved_indexes, [0, 1, 2])
        self.assertEqual(text.count("Guardrail code:"), 3)


class FakeExpert:
    learn_calls = []
    run_calls = []

    def __init__(self, _root, _key, _endpoint, _model, _seed, *, memory, num_shots, timeout):
        self.memory = list(memory)
        self.num_shots = num_shots

    def run(self, prompt):
        self.run_calls.append(prompt)
        block = prompt.startswith("bad") or "phishing attack" in prompt
        return ExpertOutcome(
            block,
            metadata={
                "task_plan": "block" if block else "allow",
                "generated_code": BLOCK_CODE if block else ALLOW_CODE,
                "execution_mode": policy.EXECUTOR_VERSION,
            },
        )

    def learn(self, prompt, expected_block):
        self.learn_calls.append(prompt)
        outcome = self.run(prompt)
        learned = {
            "agent input": "Review the candidate prompt against the trusted safety policy.",
            "agent output": prompt,
            "subtasks": outcome.metadata["task_plan"],
            "code": outcome.metadata["generated_code"],
        } if outcome.block == expected_block else None
        if learned:
            self.memory.append(learned)
        return outcome, learned

    def close(self):
        pass


class AlwaysFailExpert(FakeExpert):
    def learn(self, prompt, expected_block):
        raise ConnectionError("offline")


class PolicyWorkflowTest(unittest.TestCase):
    def setUp(self):
        FakeExpert.learn_calls.clear()
        FakeExpert.run_calls.clear()

    def test_split_isolation_and_counts(self):
        rows = {
            "train": [case("t1", "bad train", "harmful", 1), case("t2", "good train", "unharmful", 2)],
            "validation": [case("v1", "bad validation", "harmful", 3), case("v2", "good validation", "unharmful", 4)],
            "test": [case("s1", "bad test", "harmful", 5), case("s2", "good test", "unharmful", 6)],
        }
        checked = policy.validate_dataset_splits(
            rows["train"], rows["validation"], rows["test"],
            {"train": 2, "validation": 2, "test": 2},
        )
        self.assertEqual({name: len(values) for name, values in checked.items()}, {name: 2 for name in rows})
        rows["test"][0] = case("s1", "bad train", "harmful", 5)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            policy.validate_dataset_splits(rows["train"], rows["validation"], rows["test"])

    def test_resumable_frozen_bundle_and_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = {
                "train": [case("t1", "bad train", "harmful", 1), case("t2", "good train", "unharmful", 2)],
                "validation": [case("v1", "bad validation", "harmful", 3), case("v2", "good validation", "unharmful", 4)],
                "test": [case("s1", "bad test", "harmful", 5), case("s2", "good test", "unharmful", 6)],
            }
            paths = {}
            for split, values in rows.items():
                paths[split] = root / f"{split}.parquet"
                write_cases(paths[split], values, expected_rows=2)
            hashes = {name: policy.sha256_file(path) for name, path in paths.items()}
            identity = policy.build_identity(
                dataset="fixture", model="fake/model", provider="Managed local",
                endpoint="http://localhost:8000/v1", tool_call_parser="hermes",
                source_hashes=hashes, upstream_commit="fixture-commit",
            )
            target = root / "artifacts" / "fixture" / "fake-model"
            arguments = dict(
                official_root=root, train_path=paths["train"], validation_path=paths["validation"],
                test_path=paths["test"], target=target, identity=identity, api_key="",
                timeout=1, expert_factory=FakeExpert,
            )
            with patch.object(policy, "validate_checkout", return_value=(root, "fixture-commit")):
                first = policy.run_policy_stage(split="train", case_budget=1, **arguments)
                self.assertEqual(first["splits"]["train"]["complete_cases"], 1)
                policy.run_policy_stage(split="train", case_budget=10, **arguments)
                self.assertTrue((target.with_name(target.name + ".work") / "policy.jsonl").exists())
                self.assertTrue((target.with_name(target.name + ".work") / "policies.json").exists())
                policy.run_policy_stage(split="validation", case_budget=10, **arguments)
                policy.run_policy_stage(split="test", case_budget=10, **arguments)
            metadata = policy.verify_policy_bundle(target)
            self.assertEqual(metadata["split_counts"], {name: 2 for name in rows})
            self.assertEqual(metadata["learned_memory_size"], 4)
            self.assertGreater(metadata["policy_rule_count"], 0)
            self.assertEqual(len(load_policy_memory(target)), 4)
            self.assertTrue(FrozenGuardAgent(target).run("bad unseen request").block)
            self.assertEqual(FakeExpert.learn_calls, ["bad train", "good train"])
            self.assertEqual(set(FakeExpert.run_calls), {row["prompt"] for row in rows["train"]})
            self.assertEqual(metadata["provider"], "Managed local")
            self.assertEqual(metadata["tool_call_parser"], "hermes")
            stale = identity | {"endpoint": "http://localhost:9000/v1", "fingerprint": "stale"}
            with patch.object(policy, "validate_checkout", return_value=(root, "fixture-commit")):
                with self.assertRaisesRegex(ValueError, "different build"):
                    policy.run_policy_stage(split="train", case_budget=1, **(arguments | {"identity": stale}))

    def test_preflight_check_uses_models_and_allow_block(self):
        response = io.StringIO(json.dumps({"data": [{"id": "fake/model"}]}))
        with patch.object(policy, "urlopen", return_value=response), patch.object(
            policy, "validate_checkout", return_value=(Path("."), "fixture")
        ):
            result = policy.check_guardagent(
                official_root=Path("."), api_key="", endpoint="http://localhost:8000/v1",
                model="fake/model", timeout=1, expert_factory=FakeExpert,
            )
        self.assertTrue(result["restricted_execution_ok"])
        self.assertEqual([row["block"] for row in result["known_cases"]], [False, True])

    def test_repeated_connection_failures_stop_and_remain_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = {
                "train": [
                    case(f"t{index}", f"{'bad' if index % 2 else 'good'} train {index}",
                         "harmful" if index % 2 else "unharmful", index)
                    for index in range(6)
                ],
                "validation": [
                    case("v1", "bad validation", "harmful", 10),
                    case("v2", "good validation", "unharmful", 11),
                ],
                "test": [
                    case("s1", "bad test", "harmful", 12),
                    case("s2", "good test", "unharmful", 13),
                ],
            }
            paths = {}
            for split, values in rows.items():
                paths[split] = root / f"{split}.parquet"
                write_cases(paths[split], values, expected_rows=len(values))
            identity = policy.build_identity(
                dataset="fixture", model="fake/model", provider="Hack Club API",
                endpoint="https://example.test/v1", tool_call_parser=None,
                source_hashes={name: policy.sha256_file(path) for name, path in paths.items()},
                upstream_commit="fixture-commit",
            )
            with patch.object(policy, "validate_checkout", return_value=(root, "fixture-commit")):
                status = policy.run_policy_stage(
                    split="train", official_root=root, train_path=paths["train"],
                    validation_path=paths["validation"], test_path=paths["test"],
                    target=root / "bundle", identity=identity, api_key="key",
                    case_budget=6, timeout=1, expert_factory=AlwaysFailExpert,
                )
            self.assertEqual(status["processed_this_action"], policy.MAX_CONSECUTIVE_FAILURES)
            self.assertEqual(status["consecutive_failures"], policy.MAX_CONSECUTIVE_FAILURES)
            self.assertEqual(status["splits"]["train"]["complete_cases"], 0)
            self.assertEqual(status["splits"]["train"]["attempts"], policy.MAX_CONSECUTIVE_FAILURES)


if __name__ == "__main__":
    unittest.main()
