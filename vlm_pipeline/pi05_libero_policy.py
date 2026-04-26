"""pi0.5-LIBERO policy client helpers.

This module intentionally mirrors the official OpenPI LIBERO eval path used by
``/ocean/projects/cis250185p/jqian8/vlsa-aegis/main/main_aegis.py``. The policy
itself is served by ``scripts/serve_policy.py --env LIBERO`` from the vlsa-aegis
repo; this client only prepares SafeLIBERO observations and consumes action
chunks from that websocket server.
"""

import os
import sys
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np


DEFAULT_OPENPI_CLIENT_SRC = (
    "/ocean/projects/cis250185p/jqian8/vlsa-aegis/openpi/packages/openpi-client/src"
)


def _prepend_openpi_client_src(openpi_client_src: Optional[str] = None) -> Path:
    """Make the OpenPI python client from vlsa-aegis importable."""
    candidates = [
        openpi_client_src or "",
        os.environ.get("OPENPI_CLIENT_SRC", ""),
        os.environ.get("OPENPI_CLIENT_PATH", ""),
        DEFAULT_OPENPI_CLIENT_SRC,
    ]
    for raw in candidates:
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if (candidate / "openpi_client" / "__init__.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    raise FileNotFoundError(
        "Could not find OpenPI client source. Set --openpi_client_src or "
        "OPENPI_CLIENT_SRC to the directory containing openpi_client/."
    )


class Pi05LiberoPolicy:
    """Websocket-backed pi0.5-LIBERO policy with OpenPI preprocessing."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        resize_size: int = 224,
        replan_steps: int = 5,
        openpi_client_src: Optional[str] = None,
    ) -> None:
        _prepend_openpi_client_src(openpi_client_src)
        from openpi_client import image_tools  # pylint: disable=import-outside-toplevel
        from openpi_client import websocket_client_policy  # pylint: disable=import-outside-toplevel

        self.host = host
        self.port = port
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self._image_tools = image_tools
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self._action_plan = deque()

    def reset(self) -> None:
        self._action_plan.clear()
        if hasattr(self._client, "reset"):
            self._client.reset()

    def get_action(
        self,
        obs: dict,
        task_description: str,
        quat2axisangle: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        if not self._action_plan:
            action_chunk = self._infer_action_chunk(obs, task_description, quat2axisangle)
            if len(action_chunk) < self.replan_steps:
                raise ValueError(
                    f"pi0.5 server returned {len(action_chunk)} actions, "
                    f"but replan_steps={self.replan_steps}."
                )
            self._action_plan.extend(action_chunk[: self.replan_steps])

        return np.asarray(self._action_plan.popleft())

    def _infer_action_chunk(
        self,
        obs: dict,
        task_description: str,
        quat2axisangle: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

        img = self._image_tools.convert_to_uint8(
            self._image_tools.resize_with_pad(img, self.resize_size, self.resize_size)
        )
        wrist_img = self._image_tools.convert_to_uint8(
            self._image_tools.resize_with_pad(wrist_img, self.resize_size, self.resize_size)
        )

        element = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
            "prompt": str(task_description),
        }
        return np.asarray(self._client.infer(element)["actions"])
