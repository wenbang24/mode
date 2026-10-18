"""Check result reporting without loading benchmark models or marimo."""
import ast
from pathlib import Path
import unittest


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


if __name__ == "__main__":
    unittest.main()
