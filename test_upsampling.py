"""Offline contract tests for router up-sampling helpers."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from scripts import laya_router
from scripts import upsampling


def make_rows(route, count, *, category="cat-a", difficulty="hard", start=0, label="harmful"):
    return [{
        "case_id": f"{route}-{start + index}",
        "prompt": f"Prompt {route} {start + index}",
        "prompt_harm_label": label,
        "target": route,
        "category": category,
        "difficulty": difficulty,
    } for index in range(count)]


class FoldAndDifficultyTest(unittest.TestCase):
    def test_folds_are_stratified_disjoint_complete_and_deterministic(self):
        rows = []
        for route, count in zip(laya_router.ROUTES, (37, 22, 13, 8)):
            rows.extend(make_rows(route, count))
        folds = upsampling.split_folds(rows, folds=5, seed=42)
        self.assertEqual(folds, upsampling.split_folds(list(reversed(rows)), folds=5, seed=42))
        flattened = [row["case_id"] for fold in folds for row in fold]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {row["case_id"] for row in rows})
        for route in laya_router.ROUTES:
            counts = [sum(row["target"] == route for row in fold) for fold in folds]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_difficulty_boundary_and_wrong_argmax(self):
        routes = laya_router.ROUTES
        probabilities = {route: 0.1 for route in routes}
        probabilities[routes[0]] = 0.6
        self.assertEqual(upsampling.difficulty_for_prediction(routes[0], probabilities), "easy")
        probabilities[routes[0]] = 0.59
        probabilities[routes[1]] = 0.61
        self.assertEqual(upsampling.difficulty_for_prediction(routes[0], probabilities), "hard")


class CategoryAndBalanceTest(unittest.TestCase):
    def test_category_inventory_and_assignment(self):
        cases = [
            {"subcategory": "alpha, beta"},
            {"subcategory": "gamma"},
            {"subcategory": None},
        ]
        categories = upsampling.build_category_list(cases)
        self.assertEqual(categories, ["alpha", "beta", "gamma", "benign"])
        calls = []

        def llm(case, choices):
            calls.append(case["case_id"])
            return {"multi": "beta", "missing": None}.get(case["case_id"])

        self.assertEqual(upsampling.assign_category(
            {"case_id": "single", "prompt_harm_label": "harmful", "subcategory": "alpha, unknown"},
            categories, llm), "alpha")
        self.assertEqual(upsampling.assign_category(
            {"case_id": "multi", "prompt_harm_label": "harmful", "subcategory": "alpha, beta"},
            categories, llm), "beta")
        self.assertEqual(upsampling.assign_category(
            {"case_id": "missing", "prompt_harm_label": "harmful", "subcategory": None},
            categories, llm), "alpha")
        self.assertEqual(upsampling.assign_category(
            {"case_id": "benign", "prompt_harm_label": "unharmful", "subcategory": None},
            categories, lambda *_: self.fail("benign cases must not call the LLM")), "benign")
        self.assertEqual(calls, ["multi", "missing"])

    def test_category_labeling_resumes_finished_rows(self):
        cases = [
            {"case_id": "a", "prompt_harm_label": "unharmful", "subcategory": None},
            {"case_id": "b", "prompt_harm_label": "harmful", "subcategory": "alpha, beta"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "categories.jsonl"
            first = upsampling.label_categories(cases[:1], ["alpha", "benign"], path, None)
            llm_calls = []
            second = upsampling.label_categories(
                cases, ["alpha", "beta", "benign"], path,
                lambda case, _: llm_calls.append(case["case_id"]) or "beta")
            self.assertEqual(first, [{"case_id": "a", "category": "benign"}])
            self.assertEqual(second, [
                {"case_id": "a", "category": "benign"},
                {"case_id": "b", "category": "beta"},
            ])
            self.assertEqual(llm_calls, ["b"])
            self.assertEqual(len(upsampling.read_jsonl(path)), 2)

    def test_balance_downsamples_by_category_and_caps_generation_at_half(self):
        rows = []
        rows.extend(make_rows(laya_router.ROUTES[0], 6, category="alpha"))
        rows.extend(make_rows(laya_router.ROUTES[0], 6, category="beta", start=10))
        rows.extend(make_rows(laya_router.ROUTES[1], 2, category="alpha"))
        rows.extend(make_rows(laya_router.ROUTES[2], 1, category="alpha"))
        balance = upsampling.balance_cases(rows, n=4)
        first_route, second_route, third_route, empty_route = laya_router.ROUTES
        selected = [row for row in balance["kept_cases"] if row["target"] == first_route]
        category_counts = {category: sum(row["category"] == category for row in selected)
                           for category in ("alpha", "beta")}
        self.assertEqual(len(selected), 4)
        self.assertEqual(category_counts, {"alpha": 2, "beta": 2})
        self.assertEqual(balance["routes"][second_route], {"count": 2, "target": 4, "deficit": 2})
        self.assertEqual(balance["routes"][third_route], {"count": 1, "target": 2, "deficit": 1})
        self.assertEqual(balance["routes"][empty_route], {"count": 0, "target": 0, "deficit": 0})


class GenerationTest(unittest.TestCase):
    def test_allocation_and_few_shot_fallback_are_deterministic(self):
        routes = laya_router.ROUTES
        rows = [
            *make_rows(routes[0], 1, category="rare", difficulty="hard"),
            *make_rows(routes[0], 4, category="common", difficulty="easy", start=10),
            *make_rows(routes[0], 2, category="common", difficulty="hard", start=20),
        ]
        balance = {"routes": {route: {"deficit": 0} for route in routes}}
        balance["routes"][routes[0]]["deficit"] = 2
        requests = upsampling.allocate_requests(rows, balance, seed=17)
        self.assertEqual(requests, upsampling.allocate_requests(rows, balance, seed=17))
        self.assertEqual(len(requests), 4)
        self.assertEqual([row["category"] for row in requests], ["rare", "common", "rare", "common"])
        self.assertEqual(
            [row["difficulty"] for row in requests],
            ["hard", "easy", "hard", "easy"],
        )
        self.assertEqual(requests[0]["expected_label"], "harmful")
        selected = upsampling.choose_few_shot(
            rows, routes[0], "rare", "hard", request_index=0, seed=17)
        self.assertEqual([row["case_id"] for row in selected], [rows[0]["case_id"]])
        topped_up = upsampling.choose_few_shot(
            rows, routes[0], "common", "hard", request_index=2, seed=17)
        self.assertEqual(len(topped_up), 5)
        self.assertEqual(len({row["case_id"] for row in topped_up}), 5)

    def test_dedupe_checks_frozen_sets_and_prior_generated_prompts(self):
        frozen = [{"prompt": "Frozen   prompt"}]
        rows = [
            {"request_id": "1", "request_index": 0, "prompt": " frozen prompt "},
            {"request_id": "2", "request_index": 1, "prompt": "New prompt"},
            {"request_id": "3", "request_index": 2, "prompt": "new   prompt"},
            {"request_id": "4", "request_index": 3, "prompt": "  "},
        ]
        selected, dropped = upsampling.deduplicate_generated(rows, frozen)
        self.assertEqual([row["request_id"] for row in selected], ["2"])
        self.assertEqual([row["drop_reason"] for row in dropped], ["duplicate", "duplicate", "empty"])

    def test_resumable_jsonl_skips_completed_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.jsonl"
            calls = []
            handler = lambda item: calls.append(item["id"]) or {"id": item["id"], "value": item["id"] * 2}
            upsampling.resumable_jsonl([{"id": 1}, {"id": 2}], path, "id", handler)
            upsampling.resumable_jsonl([{"id": 1}, {"id": 2}, {"id": 3}], path, "id", handler)
            self.assertEqual(calls, [1, 2, 3])
            self.assertEqual(len(upsampling.read_jsonl(path)), 3)


class DriveTest(unittest.TestCase):
    def test_done_marker_upload_order_and_restore_of_complete_or_partial_steps(self):
        if importlib.util.find_spec("fsspec"):
            import fsspec

            fs = fsspec.filesystem("memory")
        else:
            fs = _FakeMemoryFS()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = root / "completed"
            (completed / "nested").mkdir(parents=True)
            (completed / "nested" / "result.jsonl").write_text('{"id": 1}\n')
            (completed / "checkpoint").mkdir()
            (completed / "checkpoint" / "weights.bin").write_bytes(b"do not upload")
            (completed / "epochs").mkdir()
            (completed / "epochs" / "checkpoint").write_bytes(b"do not upload")
            upload_order = []
            put_file = fs.put_file

            def track_upload(source, destination):
                upload_order.append(destination)
                put_file(source, destination)

            fs.put_file = track_upload
            upsampling.finish_step(completed, {"complete": True}, fs, "MoDE/upsampling/prep/completed")
            self.assertTrue(upload_order[-1].endswith("/done.json"))
            self.assertFalse(any("checkpoint" in path or "/epochs/" in path for path in upload_order))

            partial = root / "partial.jsonl"
            partial.write_text('{"request_id": "unfinished"}\n')
            fs.makedirs("MoDE/upsampling/prep/partial", exist_ok=True)
            put_file(str(partial), "MoDE/upsampling/prep/partial/progress.jsonl")
            notes = root / "notes.txt"
            notes.write_text("not resumable")
            fs.makedirs("MoDE/upsampling/prep/unfinished", exist_ok=True)
            put_file(str(notes), "MoDE/upsampling/prep/unfinished/notes.txt")

            restored = upsampling.restore_from_drive(
                fs, "MoDE/upsampling", root / "restored")
            restored_files = {Path(path).relative_to(root / "restored").as_posix() for path in restored}
            self.assertIn("prep/completed/nested/result.jsonl", restored_files)
            self.assertIn("prep/completed/done.json", restored_files)
            self.assertIn("prep/partial/progress.jsonl", restored_files)
            self.assertNotIn("prep/unfinished/notes.txt", restored_files)
            self.assertFalse((root / "restored/prep/completed/checkpoint/weights.bin").exists())


class WeightAndSummaryTest(unittest.TestCase):
    def test_b1_inverse_sqrt_weights_are_mean_one(self):
        rows = []
        for route, count in zip(laya_router.ROUTES, (1, 4, 9, 16)):
            rows.extend({
                "expected": {"expert": route},
                "state": f"prompt {route} {index}",
            } for index in range(count))
        weighted = upsampling.weighted_training_rows(rows)
        self.assertAlmostEqual(sum(row["weight"] for row in weighted) / len(weighted), 1.0)
        route_weight = {
            route: weighted[next(index for index, row in enumerate(weighted)
                                 if row["expected"]["expert"] == route)]["weight"]
            for route in laya_router.ROUTES
        }
        for route, count in zip(laya_router.ROUTES, (1, 4, 9, 16)):
            self.assertAlmostEqual(route_weight[route] * math.sqrt(count), route_weight[laya_router.ROUTES[0]])

    def test_bootstrap_ci_and_summary_metric_shapes(self):
        interval = upsampling.bootstrap_ci([0, 1, 1, 1], resamples=100, seed=0)
        self.assertEqual(interval["resamples"], 100)
        self.assertEqual(interval["confidence"], 0.95)
        self.assertLessEqual(interval["lower"], interval["estimate"])
        self.assertLessEqual(interval["estimate"], interval["upper"])

        routes = laya_router.ROUTES
        cases = [
            {"case_id": "a", "prompt_harm_label": "harmful"},
            {"case_id": "b", "prompt_harm_label": "unharmful"},
        ]
        targets = {"a": routes[0], "b": routes[1]}
        outcomes = {
            case_id: {route: {"succeeded": route == target} for route in routes}
            for case_id, target in targets.items()
        }
        predictions = {
            seed: {
                "a": {"expert": routes[0], "probabilities": {route: 0.25 for route in routes}},
                "b": {"expert": routes[1], "probabilities": {route: 0.25 for route in routes}},
            }
            for seed in (42, 43, 44)
        }
        summary = upsampling.summarize_predictions(
            cases, targets, outcomes, predictions, {"a": "alpha", "b": "benign"},
            resamples=20, seed=0)
        self.assertEqual(summary["routing_accuracy"]["estimate"], 1.0)
        self.assertIn("picked_expert_success_benign", summary)
        self.assertEqual(summary["accuracy_category:alpha"]["estimate"], 1.0)
        self.assertEqual(summary["ece"]["resamples"], 20)


class _FakeMemoryFS:
    def __init__(self):
        self.files = {}
        self.directories = set()

    def makedirs(self, path, exist_ok=False):
        self.directories.add(path.rstrip("/"))

    def put_file(self, source, destination):
        self.files[destination] = Path(source).read_bytes()
        parent = destination.rsplit("/", 1)[0] if "/" in destination else ""
        self.makedirs(parent, exist_ok=True)

    def get_file(self, source, destination):
        Path(destination).write_bytes(self.files[source])

    def exists(self, path):
        return path in self.files or path in self.directories

    def isdir(self, path):
        prefix = path.rstrip("/") + "/"
        return path in self.directories or any(name.startswith(prefix) for name in self.files)

    def ls(self, path, detail=False):
        prefix = path.rstrip("/") + "/"
        children = set()
        for name in [*self.files, *self.directories]:
            if name.startswith(prefix):
                remainder = name[len(prefix):]
                if remainder:
                    children.add(prefix + remainder.split("/", 1)[0])
        return sorted(children)


if __name__ == "__main__":
    unittest.main()
