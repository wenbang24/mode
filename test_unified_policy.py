import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import create_unified_policy

POLICY = "def classify(prompt):\n    return 'bomb' in prompt\n"


def fake_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            system = body["messages"][0]["content"]
            kind = (
                "summary" if system.startswith("You are learning")
                else "combine" if system.startswith("Consolidate")
                else "synthesis"
            )
            requests.append((kind, body["model"], self.headers.get("Authorization")))
            content = POLICY if kind == "synthesis" else f"{kind} notes"
            payload = json.dumps(
                {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


def run_build(synthesis_levels):
    _, defs = create_unified_policy.policy_engine.run()
    server, requests = fake_server()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    rows = [
        {"prompt": f"prompt {i}", "prompt_harm_label": "harmful" if i % 2 else "unharmful"}
        for i in range(16)
    ]
    synthesis = None
    if synthesis_levels is not None:
        synthesis = {
            "api_base": base,
            "model": "cloud",
            "api_key": "secret",
            "combine_levels": synthesis_levels,
        }
    try:
        source, stats = defs["build_policy"](
            rows,
            "Batch summaries + final synthesis",
            "Boolean",
            2,
            base,
            "local",
            5,
            lambda message: None,
            synthesis=synthesis,
        )
    finally:
        server.shutdown()
    assert source == POLICY.strip()
    return requests, stats


def test_local_only_by_default():
    requests, stats = run_build(None)
    # 8 summaries reduce 8 -> 4 -> 2 with fan-in 2, then one synthesis call.
    assert [kind for kind, _, _ in requests] == ["summary"] * 8 + ["combine"] * 6 + ["synthesis"]
    assert {model for _, model, _ in requests} == {"local"}
    assert stats["model_calls_by_model"] == {"local": 15}


def test_cloud_takes_top_combining_rounds_and_synthesis():
    requests, stats = run_build(1)
    assert [(kind, model) for kind, model, _ in requests] == (
        [("summary", "local")] * 8
        + [("combine", "local")] * 4
        + [("combine", "cloud")] * 2
        + [("synthesis", "cloud")]
    )
    assert all((auth == "Bearer secret") == (model == "cloud") for _, model, auth in requests)
    assert stats["model_calls_by_model"] == {"local": 12, "cloud": 3}


def test_cloud_synthesis_only_and_all_rounds():
    requests, _ = run_build(0)
    assert [model for kind, model, _ in requests if kind == "combine"] == ["local"] * 6
    assert requests[-1][:2] == ("synthesis", "cloud")
    requests, _ = run_build(5)
    assert {model for kind, model, _ in requests if kind != "summary"} == {"cloud"}
    assert {model for kind, model, _ in requests if kind == "summary"} == {"local"}
