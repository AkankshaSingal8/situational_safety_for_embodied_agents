"""Minimal, dependency-light ZMQ+msgpack client for `gr00t.policy.server_client
.PolicyServer`, reimplemented here rather than imported.

Why reimplement instead of `from gr00t.policy.server_client import
PolicyClient`: that import would pull in the full `gr00t` package (torch,
transformers, deepspeed, ...) into the `oopsieverse_robocasa` conda env,
which needs to stay a lightweight robosuite/RoboCasa-only sim env (same
reasoning as `cosmos_server.py`'s docstring on keeping heavy ML stacks out
of the sim env -- this repo's established house style). Only `pyzmq` +
`msgpack` (both tiny, no heavy deps) are needed for the wire protocol
itself.

Wire protocol reverse-engineered by reading
`gr00t/policy/server_client.py`'s `MsgSerializer`/`PolicyServer`/
`PolicyClient` classes directly (at git tag n1.6.1-release, not guessed):
ZMQ REQ/REP, msgpack-encoded, with numpy arrays encoded as
`{"__ndarray_class__": True, "as_npy": <.npy bytes>}`.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import msgpack
import numpy as np
import zmq


def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class GrootZmqClient:
    """Thin REQ-socket client matching `PolicyServer`'s endpoint protocol."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 60000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint: str, data: Optional[dict] = None) -> Any:
        request: dict = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self.socket.send(msgpack.packb(request, default=_encode))
        message = self.socket.recv()
        response = msgpack.unpackb(message, object_hook=_decode)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping")
            return True
        except zmq.error.Again:
            self._init_socket()
            return False

    def get_action(self, observation: dict, options: Optional[dict] = None):
        """Returns (action_dict, info_dict), matching Gr00tPolicy.get_action()."""
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)
