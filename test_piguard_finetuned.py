"""Offline tests for the fine-tuned PIGuard adapter."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.experts.piguard_finetuned import PIGuardArtifact


class PIGuardTokenizerTest(unittest.TestCase):
    def test_transformers_4_converts_list_extra_special_tokens(self):
        import transformers

        tokens = ["<|im_start|>", "<|im_end|>"]
        metadata = {
            "dataset": "aegis",
            "training_mode": "full",
            "pooling": "first",
            "model_id": "model",
            "max_length": 512,
            "sha256": "digest",
        }

        class StopAfterTokenizer(Exception):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_dir = root / "bundle" / "tokenizer"
            tokenizer_dir.mkdir(parents=True)
            (tokenizer_dir / "tokenizer_config.json").write_text(
                json.dumps({"extra_special_tokens": tokens}), encoding="utf-8")
            with (
                mock.patch.object(PIGuardArtifact, "inspect_artifact", return_value=metadata),
                mock.patch.object(PIGuardArtifact, "_extract_bundle", return_value=(root / "bundle", {})),
                mock.patch.object(transformers, "__version__", "4.57.6"),
                mock.patch.object(transformers.AutoTokenizer, "from_pretrained") as load_tokenizer,
                mock.patch.object(transformers.AutoModel, "from_pretrained", side_effect=StopAfterTokenizer),
            ):
                with self.assertRaises(StopAfterTokenizer):
                    PIGuardArtifact(root / "artifact.zip", "aegis", root / "cache", device="cpu")

            load_tokenizer.assert_called_once_with(
                tokenizer_dir,
                local_files_only=True,
                extra_special_tokens={},
                additional_special_tokens=tokens,
            )


if __name__ == "__main__":
    unittest.main()
