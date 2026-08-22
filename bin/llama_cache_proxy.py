#!/usr/bin/env python3
"""Cache-preserving process wrapper/proxy for llama-server.

llama-swap unloads the whole model process. This wrapper saves every configured
llama.cpp slot before the child exits and restores those slots before exposing a
healthy frontend after the next start.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}


class State:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.ready = False
        self.stopping = False
        self.child: subprocess.Popen[bytes] | None = None
        self.active = 0
        self.cv = threading.Condition()
        self.last_restore: dict[str, dict[str, Any]] = {}
        self.last_save: dict[str, dict[str, Any]] = {}

    @property
    def backend_base(self) -> str:
        return f"http://{self.args.backend_host}:{self.args.backend_port}"

    @property
    def snapshot_dir(self) -> Path:
        return Path(self.args.snapshot_dir)

    def snapshot_name(self, slot_id: int) -> str:
        return f"slot{slot_id}.bin"

    def snapshot_path(self, slot_id: int) -> Path:
        return self.snapshot_dir / self.snapshot_name(slot_id)

    def temp_snapshot_path(self, slot_id: int) -> Path:
        return self.snapshot_dir / f"{self.snapshot_name(slot_id)}.tmp"

    def metadata_path(self, slot_id: int) -> Path:
        return self.snapshot_dir / f"{self.snapshot_name(slot_id)}.meta.json"

    def request_json(self, path: str, body: dict[str, Any] | None = None, timeout: float = 60) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.backend_base + path,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
        return json.loads(raw) if raw else {}

    def wait_backend(self) -> None:
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            if self.child is not None and self.child.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup with code {self.child.returncode}")
            try:
                result = self.request_json("/health", timeout=2)
                if isinstance(result, dict) and result.get("status") == "ok":
                    return
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass
            time.sleep(0.25)
        raise TimeoutError("llama-server did not become healthy before startup timeout")

    def restore_slot(self, slot_id: int) -> None:
        path = self.snapshot_path(slot_id)
        if not path.exists():
            return
        result = self.request_json(
            f"/slots/{slot_id}?action=restore",
            {"filename": path.name},
            timeout=self.args.slot_timeout,
        )
        n_restored = int(result.get("n_restored", -1))
        expected: int | None = None
        meta_path = self.metadata_path(slot_id)
        if meta_path.exists():
            try:
                expected = int(json.loads(meta_path.read_text()).get("n_saved"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        if n_restored < 0 or (expected is not None and n_restored != expected):
            raise RuntimeError(
                f"slot {slot_id} restore verification failed: restored={n_restored}, expected={expected}"
            )
        self.last_restore[str(slot_id)] = result
        print(f"cache-wrapper: restored slot {slot_id}: {n_restored} tokens", file=sys.stderr, flush=True)

    def restore_all(self) -> None:
        for slot_id in range(self.args.slot_count):
            self.restore_slot(slot_id)

    def save_slot(self, slot_id: int) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        final = self.snapshot_path(slot_id)
        temp = self.temp_snapshot_path(slot_id)
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        result = self.request_json(
            f"/slots/{slot_id}?action=save",
            {"filename": temp.name},
            timeout=self.args.slot_timeout,
        )
        n_saved = int(result.get("n_saved", -1))
        if n_saved < 0 or not temp.exists() or temp.stat().st_size == 0:
            raise RuntimeError(f"slot {slot_id} save verification failed: {result}")
        os.replace(temp, final)
        meta = {
            "n_saved": n_saved,
            "bytes": final.stat().st_size,
            "saved_at": time.time(),
            "slot_id": slot_id,
        }
        meta_path = self.metadata_path(slot_id)
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        tmp_meta.write_text(json.dumps(meta, indent=2) + "\n")
        os.replace(tmp_meta, meta_path)
        self.last_save[str(slot_id)] = result
        print(
            f"cache-wrapper: saved slot {slot_id}: {n_saved} tokens ({meta['bytes']} bytes)",
            file=sys.stderr,
            flush=True,
        )

    def save_all(self) -> None:
        errors: list[str] = []
        for slot_id in range(self.args.slot_count):
            try:
                self.save_slot(slot_id)
            except Exception as exc:
                errors.append(f"slot {slot_id}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def transform_json(self, body: bytes, content_type: str) -> bytes:
        if "application/json" not in content_type.lower() or not body or not self.args.reasoning_budget_map:
            return body
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return body
        if not isinstance(data, dict):
            return body
        effort = data.get("reasoning_effort")
        if not isinstance(effort, str) or effort not in self.args.reasoning_budget_map:
            return body
        budget = self.args.reasoning_budget_map[effort]
        if budget is None or budget < 0:
            data.pop("thinking_budget_tokens", None)
        else:
            data["thinking_budget_tokens"] = budget
        return json.dumps(data, separators=(",", ":")).encode()


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llama-cache-wrapper/0.2"

    @property
    def state(self) -> State:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.state.args.access_log:
            super().log_message(fmt, *args)

    def send_json(self, status: int, obj: Any) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self) -> None:
        state = self.state
        if self.path == "/_wrapper/status":
            self.send_json(200, {
                "status": "ok" if state.ready and not state.stopping else "loading",
                "ready": state.ready,
                "stopping": state.stopping,
                "active_requests": state.active,
                "snapshot_dir": str(state.snapshot_dir),
                "slot_count": state.args.slot_count,
                "last_restore": state.last_restore,
                "last_save": state.last_save,
                "child_pid": None if state.child is None else state.child.pid,
            })
            return
        if not state.ready or state.stopping:
            self.send_json(503, {"status": "loading"})
            return

        content_length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(content_length) if content_length else b""
        body = state.transform_json(body, self.headers.get("Content-Type", ""))
        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in HOP_BY_HOP and k.lower() not in {"host", "content-length"}
        }
        if body:
            headers["Content-Length"] = str(len(body))

        with state.cv:
            state.active += 1
        try:
            conn = http.client.HTTPConnection(
                state.args.backend_host, state.args.backend_port, timeout=state.args.proxy_timeout
            )
            conn.request(self.command, self.path, body=body or None, headers=headers)
            response = conn.getresponse()
            self.send_response(response.status, response.reason)
            has_length = response.getheader("Content-Length") is not None
            for key, value in response.getheaders():
                lk = key.lower()
                if lk in HOP_BY_HOP or lk == "content-length":
                    continue
                self.send_header(key, value)
            if has_length:
                self.send_header("Content-Length", response.getheader("Content-Length"))
            else:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                if has_length:
                    self.wfile.write(chunk)
                else:
                    self.wfile.write(f"{len(chunk):X}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                self.wfile.flush()
            if not has_length:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            conn.close()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            if not self.wfile.closed:
                print(f"cache-wrapper: proxy error for {self.path}: {exc}", file=sys.stderr, flush=True)
        finally:
            with state.cv:
                state.active -= 1
                state.cv.notify_all()

    do_GET = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_DELETE = _proxy
    do_OPTIONS = _proxy


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], state: State) -> None:
        self.state = state
        super().__init__(addr, ProxyHandler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exception()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def parse_reasoning_map(raw: str | None) -> dict[str, int | None]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("reasoning budget map must be a JSON object")
    out: dict[str, int | None] = {}
    for key, budget in value.items():
        if budget is None:
            out[str(key)] = None
        elif isinstance(budget, int):
            out[str(key)] = budget
        else:
            raise argparse.ArgumentTypeError("reasoning budget values must be integers or null")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--listen-host", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, required=True)
    p.add_argument("--backend-host", default="127.0.0.1")
    p.add_argument("--backend-port", type=int, required=True)
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--slot-count", type=int, default=1)
    p.add_argument("--startup-timeout", type=float, default=300)
    p.add_argument("--slot-timeout", type=float, default=1800)
    p.add_argument("--drain-timeout", type=float, default=300)
    p.add_argument("--proxy-timeout", type=float, default=None)
    p.add_argument("--access-log", action="store_true")
    p.add_argument(
        "--reasoning-budget-map",
        type=parse_reasoning_map,
        default={},
        help='optional JSON map, e.g. {"low":2048,"medium":8192,"high":32768,"xhigh":-1}',
    )
    p.add_argument("server_command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.slot_count < 1:
        p.error("--slot-count must be >= 1")
    if args.server_command and args.server_command[0] == "--":
        args.server_command = args.server_command[1:]
    if not args.server_command:
        p.error("llama-server command is required after --")
    return args


def main() -> int:
    args = parse_args()
    state = State(args)
    state.snapshot_dir.mkdir(parents=True, exist_ok=True)
    command = list(args.server_command) + [
        "--host", args.backend_host,
        "--port", str(args.backend_port),
        "--slot-save-path", str(state.snapshot_dir),
    ]
    state.child = subprocess.Popen(command, start_new_session=True)
    proxy = ProxyServer((args.listen_host, args.listen_port), state)
    proxy_thread = threading.Thread(target=proxy.serve_forever, name="proxy", daemon=True)
    proxy_thread.start()
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        state.stopping = True
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    exit_code = 0
    try:
        state.wait_backend()
        state.restore_all()
        state.ready = True
        print(
            f"cache-wrapper: ready on {args.listen_host}:{args.listen_port}; restored up to {args.slot_count} slot(s)",
            file=sys.stderr,
            flush=True,
        )
        while not stop_event.wait(0.25):
            if state.child.poll() is not None:
                raise RuntimeError(f"llama-server exited unexpectedly with code {state.child.returncode}")
    except Exception as exc:
        print(f"cache-wrapper: fatal: {exc}", file=sys.stderr, flush=True)
        exit_code = 1
        state.stopping = True
    finally:
        state.ready = False
        state.stopping = True
        proxy.shutdown()
        proxy.server_close()
        deadline = time.monotonic() + args.drain_timeout
        with state.cv:
            while state.active and time.monotonic() < deadline:
                state.cv.wait(timeout=max(0.01, min(0.5, deadline - time.monotonic())))
        if state.active:
            print(f"cache-wrapper: drain timeout with {state.active} active request(s)", file=sys.stderr, flush=True)
            exit_code = 1
        elif state.child is not None and state.child.poll() is None:
            try:
                state.save_all()
            except Exception as exc:
                print(f"cache-wrapper: failed to save cache: {exc}", file=sys.stderr, flush=True)
                exit_code = 1
        if state.child is not None and state.child.poll() is None:
            try:
                os.killpg(state.child.pid, signal.SIGTERM)
                state.child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(state.child.pid, signal.SIGKILL)
                state.child.wait(timeout=5)
        proxy_thread.join(timeout=2)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
