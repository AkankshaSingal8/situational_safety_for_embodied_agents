"""Minimal server wrapping the off-the-shelf Cosmos-Policy-RoboCasa checkpoint.

Run inside cosmos-policy's existing Apptainer container + .venv_rhel8 (see
../slurm/launch_cosmos_server via apptainer exec) -- NOT a fresh env, this
reuses the already-built, already-debugged container/venv read-only.

Deliberately uses a stdlib-only, length-prefixed pickle-over-TCP protocol
(socket + pickle + struct) instead of FastAPI/uvicorn/json-numpy like the
OpenVLA server: none of those packages are installed in .venv_rhel8, and
this repo's own history shows that env is fragile (glibc/LD_LIBRARY_PATH
segfault risk, see the risk-gate spike's docstring) -- adding new pip
packages to it is an avoidable risk for zero benefit when a stdlib-only
protocol does the same job. Pickle is safe here: both processes are
same-machine, same-user, trusted, and numpy versions match closely enough
(cosmos venv: numpy 2.2.6, oopsieverse_robocasa: numpy 2.2.5) that array
pickling round-trips cleanly.

Run:
    (inside the cosmos-policy apptainer/.venv_rhel8, see
    ../slurm/spike_cosmos_risk_gate.slurm for the exact apptainer invocation)
    python oopsieverse_eval/servers/cosmos_server.py --port 8400

Request/response wire format: 4-byte big-endian length prefix + pickle
payload. Request dict:
    {"primary_image": <H,W,3 uint8>, "secondary_image": <H,W,3 uint8>,
     "wrist_image": <H,W,3 uint8>, "proprio": <9,> float32, "instruction": str}
Response dict:
    {"action": <T,7> float32} on success, or {"error": "<traceback str>"}.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import socket
import struct
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.join(_HERE, "..", "eval")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from cosmos_config import CosmosEvalConfig, DEFAULT_COSMOS_CONFIG_KWARGS  # noqa: E402

COSMOS_REPO = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/cosmos-policy"
if COSMOS_REPO not in sys.path:
    sys.path.insert(0, COSMOS_REPO)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cosmos_server")


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while reading")
        buf += chunk
    return buf


def recv_msg(conn: socket.socket) -> dict:
    (length,) = struct.unpack(">I", recv_exact(conn, 4))
    return pickle.loads(recv_exact(conn, length))


def send_msg(conn: socket.socket, obj: dict) -> None:
    payload = pickle.dumps(obj)
    conn.sendall(struct.pack(">I", len(payload)) + payload)


class CosmosServer:
    def __init__(self, seed: int = 195):
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_action,
            get_model,
            init_t5_text_embeddings_cache,
            load_dataset_stats,
        )

        self._get_action = get_action
        self.seed = seed
        self.cfg = CosmosEvalConfig(**DEFAULT_COSMOS_CONFIG_KWARGS)

        logger.info("Loading model from %s (config=%s)...", self.cfg.ckpt_path, self.cfg.config)
        self.model, _ = get_model(self.cfg)
        logger.info("Model loaded.")

        logger.info("Loading dataset stats from %s...", self.cfg.dataset_stats_path)
        self.dataset_stats = load_dataset_stats(self.cfg.dataset_stats_path)

        # Empty cache -> live T5 compute for any OopsieVerse instruction not
        # in Cosmos's own 24-task training vocabulary (confirmed safe/working
        # by cosmos_risk_gate_spike.py).
        init_t5_text_embeddings_cache(None)
        logger.info("Server ready.")

    def predict(self, obs: dict, instruction: str):
        result = self._get_action(
            self.cfg,
            self.model,
            self.dataset_stats,
            obs,
            instruction,
            seed=self.seed,
            num_denoising_steps_action=self.cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )
        return result["actions"]


def serve(server: CosmosServer, host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(5)  # defense-in-depth backlog; the real fix is the client's long timeout
        logger.info("Listening on %s:%d", host, port)
        while True:
            conn, addr = sock.accept()
            # Every step of handling one connection -- including the
            # error-reporting send_msg -- must never let an exception
            # escape this loop iteration. A single malformed/probe
            # connection (e.g. a bare TCP connect+close with no payload,
            # or a real client that already gave up after a slow request
            # and closed its socket) previously killed the ENTIRE server
            # process here: the fallback `send_msg(conn, {"error": ...})`
            # in the except clause was itself unguarded, so if the peer was
            # already gone, THAT send raised too, propagating out of
            # `serve()` and taking down every future request with it. This
            # is why the smoke test failed on every episode after the
            # first hiccup -- observed directly: the server log ends
            # exactly at the first ConnectionError, no further requests
            # were ever served.
            try:
                with conn:
                    try:
                        request = recv_msg(conn)
                        if request.get("_health_check"):
                            send_msg(conn, {"status": "ok"})
                            continue
                        obs = {
                            "primary_image": request["primary_image"],
                            "secondary_image": request["secondary_image"],
                            "wrist_image": request["wrist_image"],
                            "proprio": request["proprio"],
                        }
                        action = server.predict(obs, request["instruction"])
                        send_msg(conn, {"action": action})
                    except Exception:
                        logger.exception("Inference failed")
                        try:
                            send_msg(conn, {"error": traceback.format_exc()})
                        except Exception:
                            logger.warning("Could not send error response -- peer already gone.")
            except Exception:
                logger.exception("Unexpected error handling connection from %s -- continuing to serve.", addr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--seed", type=int, default=195)
    args = parser.parse_args()

    server = CosmosServer(seed=args.seed)
    serve(server, args.host, args.port)


if __name__ == "__main__":
    main()
