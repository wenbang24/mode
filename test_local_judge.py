"""CPU-only checks: uv run --with marimo --with numpy --with pyarrow python -m unittest test_local_judge"""
import ast
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from scripts.local_judge import (
    LocalJudgeServer, judge_key, normalize_endpoint, positive_timeout, request_headers,
)


def piguard_functions():
    path = Path(__file__).with_name("train_custom_piguard.py")
    cell = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef))
    body = [node for node in cell.body if not isinstance(node, ast.Return)]
    namespace = {"__file__": str(path)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class EndpointTest(unittest.TestCase):
    def test_urls_credentials_and_timeout(self):
        for host in ("localhost", "127.0.0.1", "[::1]"):
            endpoint = f"http://{host}:8000/v1"
            self.assertEqual(normalize_endpoint(endpoint + "/"), endpoint)
            with patch.dict(os.environ, {"AI_API_KEY": "hosted"}, clear=True):
                self.assertEqual(judge_key(endpoint, "AI_API_KEY"), "")
            with patch.dict(os.environ, {"JUDGE_API_KEY": "local"}, clear=True):
                self.assertEqual(judge_key(endpoint, "AI_API_KEY"), "local")
        for value in ("http://remote/v1", "https://a:bad/v1", "https://u:p@a/v1", "https://a/v1?q=x", "https://a/v1#x"):
            with self.assertRaises(ValueError):
                normalize_endpoint(value)
        for legacy in ("AI_API_KEY", "HACKCLUB_API_KEY"):
            with patch.dict(os.environ, {legacy: "old"}, clear=True):
                self.assertEqual(judge_key("https://provider/v1", legacy), "old")
            with patch.dict(os.environ, {legacy: "old", "JUDGE_API_KEY": "new"}, clear=True):
                self.assertEqual(judge_key("https://provider/v1", legacy), "new")
        self.assertNotIn("Authorization", request_headers(""))
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                positive_timeout(value)

    def test_adasteer_local_request(self):
        from scripts.experts.adasteer import AdaSteer
        context = SimpleNamespace(api_base="http://localhost:8000/v1", api_key="", judge_model="custom", seed=42, judge_timeout=999)
        def respond(request, timeout):
            self.assertEqual(timeout, 999)
            self.assertIsNone(request.get_header("Authorization"))
            self.assertEqual(json.loads(request.data)["model"], "custom")
            self.assertNotIn("structured_outputs", json.loads(request.data))
            return io.StringIO(json.dumps({"choices": [{"message": {"content": "Yes"}}]}))
        with patch("scripts.experts.adasteer.urlopen", respond):
            self.assertTrue(AdaSteer._paper_judge(context, "hi", "no", "refusal")["result"])

    def test_managed_judge_constrains_labels_without_changing_prompt(self):
        from scripts.experts.adasteer import AdaSteer
        from scripts.experts.adasteer_bundle import COMPLIANCE_PROMPT
        context = SimpleNamespace(api_base="http://localhost:8000/v1", api_key="", judge_model="custom",
                                  seed=42, constrain_judgments=True)
        def respond(request, timeout):
            body = json.loads(request.data)
            self.assertEqual(body["structured_outputs"]["choice"],
                             ["1_full_compliance", "2_full_refusal", "3_partial_refusal"])
            self.assertEqual(body["messages"], [{"role": "user", "content": COMPLIANCE_PROMPT.format(question="hi", answer="hello")}])
            return io.StringIO(json.dumps({"choices": [{"message": {"content": "1_full_compliance"}}]}))
        with patch("scripts.experts.adasteer.urlopen", respond):
            self.assertEqual(AdaSteer._paper_judge(context, "hi", "hello", "compliance", 1)["result"], "1_full_compliance")

    def test_piguard_transport_and_existing_checks(self):
        ns = piguard_functions()
        self.assertTrue(ns["self_check"]())
        def respond(request, timeout):
            self.assertEqual(timeout, 555)
            self.assertIsNone(request.get_header("Authorization"))
            self.assertEqual(json.loads(request.data)["model"], "custom")
            return io.BytesIO(b'{"choices":[{"message":{"content":"[]"}}]}')
        with patch("urllib.request.urlopen", respond):
            ns["_ai_transport"]("http://localhost:8000/v1", "chat/completions", "", {"model": "custom"}, 555)
        with self.assertRaises(ValueError):
            ns["validate_ai_model"]("missing", "", {"used": 0, "limit": 2}, transport=lambda *a: {"data": [{"id": "other"}]})
        error = HTTPError("url", 401, "denied", {}, io.BytesIO(b"denied"))
        self.addCleanup(error.close)
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ns["AIAPIError"]):
                ns["_ai_transport"]("http://localhost:8000/v1", "models", "")

    def test_adasteer_fingerprint_endpoint(self):
        from scripts.experts import adasteer_bundle as bundle
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {split: root / (split + ".parquet") for split in ("train", "validation", "test")}
            for path in paths.values():
                path.write_bytes(b"dataset")
            geometry = {"num_hidden_layers": 2, "hidden_size": 4}
            preflight = (root, "commit", paths, {split: [] for split in paths},
                         SimpleNamespace(max_position_embeddings=8192), None)
            args = dict(official_root=root, train_path=paths["train"], validation_path=paths["validation"],
                        test_path=paths["test"], output_root=root / "out", judge_refusal=lambda *a: False,
                        judge_compliance=lambda *a: "1_full_compliance", judge_model="test/model")
            with patch.object(bundle, "_runtime_preflight", return_value=preflight), patch.object(
                bundle, "validate_model_config", return_value=geometry
            ), patch.object(bundle, "_label_unsteered", side_effect=RuntimeError("stop before GPU")):
                with self.assertRaisesRegex(RuntimeError, "stop before GPU"):
                    bundle.build_bundle(**args, judge_endpoint="http://localhost:8000/v1/")
                saved = next((root / "out").rglob("build.json"))
                identity = json.loads(saved.read_text())
                self.assertEqual(identity["judge_endpoint"], "http://localhost:8000/v1")
                self.assertNotIn("api_key", saved.read_text())
                with self.assertRaisesRegex(ValueError, "stale"):
                    bundle.build_bundle(**args, judge_endpoint="http://localhost:8000/v1", judge_constrained=True)
                with self.assertRaisesRegex(ValueError, "stale"):
                    bundle.build_bundle(**args, judge_endpoint="http://localhost:8001/v1")
                with self.assertRaisesRegex(ValueError, "stale"):
                    bundle.build_bundle(**(args | {"judge_model": "other/model"}), judge_endpoint="http://localhost:8000/v1")

    def test_server_panel_does_not_start_on_render(self):
        import marimo as mo
        from scripts.local_judge import server_panel
        server = LocalJudgeServer()
        with patch.object(server, "start") as start:
            settings, _, status = server_panel(mo, server)
            self.assertEqual(settings.value["port"], 8000)
            self.assertEqual(status()["state"], "stopped")
            server_panel(mo, server)
            start.assert_not_called()


class LauncherTest(unittest.TestCase):
    def test_native_sampler_does_not_require_cuda_compiler(self):
        server = LocalJudgeServer()
        try:
            with patch("scripts.local_judge.sys.platform", "linux"), patch(
                "scripts.local_judge.shutil.which", return_value="uv"
            ), patch("scripts.local_judge.socket.socket"), patch(
                "scripts.local_judge.subprocess.Popen"
            ) as launch, patch.object(server, "status", return_value={"state": "starting"}), patch.dict(
                os.environ, {}, clear=True
            ):
                server.start("test/model")
                self.assertEqual(launch.call_args.kwargs["env"].get("VLLM_USE_FLASHINFER_SAMPLER"), "0")
        finally:
            server.process = None  # Mock process, not an owned OS process.

    def test_lifecycle(self):
        # Stand in for vLLM with an HTTP server whose worker ignores SIGTERM.
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "server.py"
            script.write_text('''import http.server, json, signal, subprocess, sys
worker = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"])
print("worker", worker.pid, flush=True)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({"data":[{"id":"test/model"}]}).encode())
    def log_message(self, *args): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
''')
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]
            server = LocalJudgeServer()
            real_popen = subprocess.Popen
            commands = []
            def fake_popen(command, **kwargs):
                commands.append(command)
                return real_popen([sys.executable, str(script), str(port)], **kwargs)
            try:
                with patch("scripts.local_judge.sys.platform", "linux"), patch("scripts.local_judge.shutil.which", return_value="uv"), patch("scripts.local_judge.subprocess.Popen", fake_popen):
                    server.start("test/model", port)
                    for _ in range(100):
                        if server.status()["state"] == "ready": break
                        time.sleep(0.05)
                    self.assertEqual(server.status()["state"], "ready")
                    server.require_ready(server.settings | {"model": " test/model "})
                    server.start("test/model", port)
                    self.assertEqual(len(commands), 1)
                    self.assertIn("--from", commands[0])
                    self.assertIn("--max-num-seqs", commands[0])
                    with self.assertRaisesRegex(RuntimeError, "Stop"):
                        server.start("other/model", port)
                    other = LocalJudgeServer()
                    with self.assertRaisesRegex(RuntimeError, "occupied"):
                        other.start("test/model", port)
                worker = int(server.status()["log"].splitlines()[0].split()[1])
                server.stop()
                self.assertEqual(server.status()["state"], "stopped")
                # A zombie is dead too; only a live worker constitutes a leak.
                for _ in range(100):
                    state = subprocess.run(["ps", "-o", "stat=", "-p", str(worker)], capture_output=True, text=True).stdout.strip()
                    if not state or state.startswith("Z"): break
                    time.sleep(0.05)
                self.assertTrue(not state or state.startswith("Z"), state)
            finally:
                server.stop()

    def test_startup_failure_and_readiness(self):
        server = LocalJudgeServer()
        try:
            server.settings = dict(model="test", port=8000, gpu_memory=.5, context_length=8192)
            server.process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"], start_new_session=True)
            server.process.wait()
            self.assertEqual(server.status()["state"], "failed")
            with self.assertRaises(RuntimeError):
                server.require_ready(server.settings)
            with self.assertRaisesRegex(RuntimeError, "exited"):
                server.start("test")
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
