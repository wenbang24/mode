"""Local judge endpoint configuration and a notebook-owned vLLM process."""
from __future__ import annotations

import atexit
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-14B-Instruct"
VLLM_VERSION = "0.27.0"


def normalize_endpoint(value):
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("Invalid judge API port") from exc
    if (not parsed.hostname or parsed.scheme not in {"http", "https"}
        or (parsed.scheme == "http" and not is_loopback(value))
        or parsed.username or parsed.password or parsed.params or parsed.query
        or parsed.fragment or any(c.isspace() for c in value)):
        raise ValueError("Judge API URL must use HTTPS (or HTTP on localhost), without credentials, query, or fragment")
    return value


def is_loopback(value):
    return urlparse(value).hostname in {"localhost", "127.0.0.1", "::1"}


def judge_key(endpoint, legacy_name):
    key = os.environ.get("JUDGE_API_KEY", "").strip()
    if not key and not is_loopback(endpoint):
        key = os.environ.get(legacy_name, "").strip()
    if not key and not is_loopback(endpoint):
        raise ValueError(f"Set JUDGE_API_KEY (or {legacy_name}) for this hosted endpoint")
    return key


def request_headers(key):
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}


def positive_timeout(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Request timeout must be positive and finite")
    return value


class LocalJudgeServer:
    def __init__(self):
        self.process = None
        self.settings = None
        self.log_path = None
        atexit.register(self.stop)

    def start(self, model, port=8000, gpu_memory=0.5, context_length=8192):
        model = model.strip()
        if not model or model.startswith("-"):
            raise ValueError("Enter a Hugging Face model ID or path")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Port must be an integer from 1 to 65535")
        if not 0 < gpu_memory < 1:
            raise ValueError("GPU memory fraction must be between 0 and 1")
        if type(context_length) is not int or context_length < 1:
            raise ValueError("Context length must be a positive integer")
        settings = dict(model=model, port=port, gpu_memory=gpu_memory, context_length=context_length)
        if self.process is not None:
            if self.settings != settings:
                raise RuntimeError("Stop the server before changing its settings")
            if self.process.poll() is None:
                return self.status()
            raise RuntimeError("Server exited; check logs, then Stop before restarting")
        if sys.platform != "linux" or not shutil.which("nvidia-smi"):
            raise RuntimeError("Managed vLLM requires a Linux NVIDIA GPU environment such as molab")
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("uv is required to start the isolated server")
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is occupied; choose another port or use Existing endpoint") from exc
        self.settings = settings
        fd, log = tempfile.mkstemp(prefix="mode-vllm-", suffix=".log")
        self.log_path = Path(log)
        env = os.environ.copy()
        # Molab has CUDA runtime libraries but no nvcc for FlashInfer sampler JIT.
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        # Never inherit server authentication from an unrelated vLLM deployment.
        for name in ("VLLM_API_KEY", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(name, None)
        command = [uv, "tool", "run", "--python", "3.12", "--from", f"vllm=={VLLM_VERSION}",
                   "vllm", "serve", model, "--host", "127.0.0.1", "--port", str(port),
                   "--served-model-name", model, "--gpu-memory-utilization", str(gpu_memory),
                   "--max-model-len", str(context_length), "--max-num-seqs", "4",
                   "--tensor-parallel-size", "1"]
        with os.fdopen(fd, "wb") as log_file:
            self.process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
                                            env=env, start_new_session=True)
        return self.status()

    def status(self):
        result = {"state": "stopped", "model": (self.settings or {}).get("model"),
                  "vllm_version": VLLM_VERSION, "log": ""}
        if self.log_path and self.log_path.exists():
            with self.log_path.open("rb") as handle:
                handle.seek(max(0, self.log_path.stat().st_size - 8000))
                result["log"] = handle.read().decode(errors="replace")
            for name in ("HF_TOKEN", "JUDGE_API_KEY", "AI_API_KEY", "HACKCLUB_API_KEY"):
                if os.environ.get(name):
                    result["log"] = result["log"].replace(os.environ[name], "[redacted]")
        if self.process is None:
            return result
        result["exit_code"] = self.process.poll()
        result["state"] = "failed" if result["exit_code"] is not None else "starting"
        result["endpoint"] = f"http://127.0.0.1:{self.settings['port']}/v1"
        if result["state"] == "starting":
            try:
                with urlopen(result["endpoint"].removesuffix("/v1") + "/health", timeout=1):
                    pass
                with urlopen(result["endpoint"] + "/models", timeout=1) as response:
                    models = json.load(response)
                if self.settings["model"] in [row.get("id") for row in models.get("data", [])]:
                    result["state"] = "ready"
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        return result

    def require_ready(self, settings):
        settings = settings | {"model": settings["model"].strip()}
        if self.settings != settings or self.status()["state"] != "ready":
            raise RuntimeError("Start the local server with the selected settings and check that it is ready")

    def stop(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # The leader may exit before its GPU workers do.
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
            except ProcessLookupError:
                pass
            self.process = None
        return self.status()


def server_panel(mo, server):
    """Callbacks own process actions; reactive notebook evaluation only reads state."""
    settings = mo.ui.dictionary({
        "model": mo.ui.text(value=DEFAULT_LOCAL_MODEL, label="Local Hugging Face model ID or path"),
        "port": mo.ui.number(start=1, stop=65535, value=8000, step=1, label="Port"),
        "gpu_memory": mo.ui.number(start=0.05, stop=0.95, value=0.5, step=0.05, label="GPU memory fraction"),
        "context_length": mo.ui.number(start=1, value=8192, step=1, label="Context length"),
    })
    get_status, set_status = mo.state(server.status())

    def action(name):
        try:
            result = server.start(**settings.value) if name == "start" else getattr(server, name)()
        except Exception as exc:
            result = server.status() | {"error": str(exc)}
        set_status(result)

    view = mo.vstack([settings, mo.hstack([
        mo.ui.button(label="Start local server", on_click=lambda _: action("start")),
        mo.ui.button(label="Check status", on_click=lambda _: action("status")),
        mo.ui.button(label="Stop server", on_click=lambda _: action("stop")),
    ]), mo.md("First start installs vLLM and downloads weights. Check status for progress. "
              "The server shares GPU memory with training; reduce model size or memory allocation if needed. "
              "Caches may not survive a molab session restart.")])
    return settings, view, get_status
