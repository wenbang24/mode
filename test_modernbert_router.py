"""Router checks and tiny offline ModernBERT training; never download model weights."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import laya_router as lr
from scripts import modernbert_router as mr
from test_laya_router import make_benchmark, make_cases


class SettingsTest(unittest.TestCase):
    def test_settings_and_label_order(self):
        self.assertEqual(mr.LABEL2ID, {"adasteer": 0, "piguard": 1, "guardagent": 2})
        self.assertEqual(mr.validate_settings({}), mr.DEFAULT_SETTINGS)
        for key, value in (("epochs", True), ("micro_batch", 0), ("grad_accum", 1.5),
                           ("learning_rate", 0), ("learning_rate", float("nan")),
                           ("warmup_ratio", 1.1), ("weight_decay", -1), ("amp", "yes")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                mr.validate_settings({key: value})

    def test_unicode_separators_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            row = {"state": "hello\u2028world\u0085again", "expected": {"expert": "piguard"}}
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(mr.read_labeled(path), [row])
            for invalid in ({**row, "state": " "}, {**row, "expected": {"expert": "missing"}}):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    mr.read_labeled(path)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "transformers", "accelerate")),
                     "Install the notebook dependencies to run the tiny offline model checks")
class TinyModernBERTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import ModernBertConfig, ModernBertForSequenceClassification, PreTrainedTokenizerFast

        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.base = cls.root / "base"
        vocabulary = {word: index for index, word in enumerate(
            ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "train", "test", "prompt", "hello", "world", "again"]
            + [str(i) for i in range(20)])}
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        backend.post_processor = TemplateProcessing(single="[CLS] $A [SEP]",
                                                    special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                              cls_token="[CLS]", sep_token="[SEP]", unk_token="[UNK]")
        config = ModernBertConfig(
            vocab_size=len(vocabulary), hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128,
            local_attention=16, global_attn_every_n_layers=2,
            pad_token_id=0, cls_token_id=1, sep_token_id=2,
            id2label=mr.ID2LABEL, label2id=mr.LABEL2ID, reference_compile=False,
            classifier_dropout=0.0, attn_implementation="sdpa",
        )
        torch.manual_seed(42)
        model = ModernBertForSequenceClassification(config)
        cls.initial_embedding = model.model.embeddings.tok_embeddings.weight.detach().clone()
        model.save_pretrained(cls.base)
        cls.tokenizer.save_pretrained(cls.base)
        cls.parquet = cls.root / "unseen.parquet"
        cls.test_parquet = cls.root / "test.parquet"
        cls.parquet.write_bytes(b"unseen fixture")
        cls.test_parquet.write_bytes(b"test fixture")
        cls.cases = make_cases()
        cls.benchmark = cls.root / "benchmark"
        make_benchmark(cls.benchmark, cls.cases, cls.parquet)
        with patch.object(lr, "read_cases", side_effect=[cls.cases, make_cases(prefix="test")]):
            cls.prepared = mr.prepare_data(cls.benchmark, cls.parquet, cls.test_parquet, cls.root / "artifacts")
        cls.settings = {**mr.DEFAULT_SETTINGS, "epochs": 2, "micro_batch": 4, "grad_accum": 2, "max_len": 8}
        cls.trained = mr.train_router(cls.prepared, cls.base, cls.root / "artifacts", cls.settings, "cpu")

    def test_training_updates_encoder_and_selects_best_checkpoint(self):
        import torch
        from transformers import AutoModelForSequenceClassification

        directory = Path(self.trained["directory"])
        self.assertTrue(math.isfinite(self.trained["summary"]["best_validation_loss"]))
        losses = [row["eval_loss"] for row in self.trained["summary"]["history"] if "eval_loss" in row]
        self.assertEqual(self.trained["summary"]["best_validation_loss"], min(losses))
        self.assertEqual(self.trained["inspection"]["total_updates"], 4)
        model = AutoModelForSequenceClassification.from_pretrained(directory / "checkpoint", local_files_only=True)
        self.assertFalse(torch.equal(self.initial_embedding, model.model.embeddings.tok_embeddings.weight))
        self.assertEqual(model.config.id2label, mr.ID2LABEL)
        self.assertEqual(model.config.modernbert_router["max_len"], 8)
        self.assertEqual(self.trained["base_revision"], mr.BASE_REVISION)
        self.assertIn("transformers", self.trained["versions"])
        policies = {row["policy"] for row in self.trained["validation"]["summary"]}
        self.assertEqual(policies, {*mr.EXPERTS, "finetuned_modernbert", "fastest_successful_oracle"})

    def test_encoding_contains_only_prompt_and_fixed_labels(self):
        rows = [{"state": "hello " * 20, "questions": {"secret": "never encode"},
                 "expected": {"expert": expert}} for expert in mr.EXPERTS]
        encoded = mr.encode_rows(rows, self.tokenizer, 8)
        self.assertEqual([row["labels"] for row in encoded], [0, 1, 2])
        self.assertEqual(len(encoded[0]["input_ids"]), 8)
        self.assertEqual(encoded[0]["input_ids"][-1], self.tokenizer.sep_token_id)
        self.assertEqual(set(encoded[0]), {"input_ids", "attention_mask", "labels"})
        self.assertEqual(encoded[0]["input_ids"], encoded[1]["input_ids"])
        with self.assertRaisesRegex(ValueError, "context"):
            mr.inspect_training(self.prepared["directory"], self.base, {"max_len": 129})

    def test_predictions_truncation_invalid_prompts_and_metadata(self):
        checkpoint = Path(self.trained["directory"]) / "checkpoint"
        with mr.ModernBERTRouter(checkpoint, "cpu") as router:
            for prompt, truncated in (("hello", False), ("hello " * 20, True)):
                result = router.predict(prompt)
                self.assertEqual(result["truncated"], truncated)
                self.assertEqual(set(result["probabilities"]), set(mr.EXPERTS))
                self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=6)
                self.assertEqual(result["confidence"], result["probabilities"][result["expert"]])
            for invalid in (None, "", " ", 123):
                with self.assertRaises(ValueError):
                    router.predict(invalid)
            # Inference must use exactly the same truncation as training.
            with patch.object(router.model, "forward", wraps=router.model.forward) as forward:
                router.predict("hello " * 20)
                self.assertEqual(forward.call_args.kwargs["input_ids"][0].tolist(),
                                 mr.encode_rows([{"state": "hello " * 20, "expected": {"expert": "adasteer"}}],
                                                self.tokenizer, 8)[0]["input_ids"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            router.predict("hello")
        with self.assertRaisesRegex(ValueError, "schema"):
            mr.ModernBERTRouter(self.base, "cpu")
        broken = self.root / "bad_config"
        broken.mkdir(exist_ok=True)
        config = json.loads((checkpoint / "config.json").read_text())
        for change in ({"id2label": {"0": "wrong", "1": "piguard", "2": "guardagent"}},
                       {"modernbert_router": {"schema_version": 1, "max_len": 1000}}):
            mr.write_json(broken / "config.json", {**config, **change})
            with self.assertRaises(ValueError):
                mr.ModernBERTRouter(broken, "cpu")

    def test_prepared_data_tampering_is_rejected(self):
        changed = copy.deepcopy(self.prepared)
        changed["manifest"]["seed"] = 99
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            mr.train_router(changed, self.base, self.root, self.settings, "cpu")
        path = Path(self.prepared["directory"]) / "train.jsonl"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "Prepared data changed"):
                mr.train_router(self.prepared, self.base, self.root, self.settings, "cpu")
        finally:
            path.write_bytes(original)

    def test_reserved_test_checks_and_export_are_portable(self):
        cases = make_cases(prefix="test")
        test_run = self.root / "test_benchmark"
        make_benchmark(test_run, cases, self.test_parquet, split="test")
        with patch.object(lr, "read_cases", return_value=cases):
            report = mr.evaluate_test(self.trained["directory"], test_run, self.test_parquet, "cpu")
        self.assertEqual(report["summary"][0]["cases"], 20)
        training_path = Path(self.trained["directory"]) / "training_data/train_cases.json"
        original = training_path.read_bytes()
        try:
            mr.write_json(training_path, [*self.cases, cases[0]])
            with patch.object(lr, "read_cases", return_value=cases):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    mr.evaluate_test(self.trained["directory"], test_run, self.test_parquet, "cpu")
        finally:
            training_path.write_bytes(original)
        export_root = self.root / "portable"
        exported = Path(mr.export_router(self.trained["directory"], export_root, "cpu"))
        with self.assertRaises(FileExistsError):
            mr.export_router(self.trained["directory"], export_root, "cpu")
        relocated = self.root / "relocated_export"
        shutil.move(exported, relocated)
        with mr.ModernBERTRouter(relocated, "cpu") as router:
            prediction = router.predict("Explain how rainbows form.")
        self.assertEqual(prediction, json.loads((relocated / "reload_check.json").read_text()))
        self.assertTrue((relocated / "training_data/manifest.json").is_file())
        self.assertTrue((relocated / "test_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
