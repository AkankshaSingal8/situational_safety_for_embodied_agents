# vlm_pipeline/dreamzero_client.py
"""
WebSocket client for DreamZero WAM inference server.

Protocol: raw WebSocket + msgpack-numpy binary serialization.
Server entry point: socket_test_optimized_AR.py in dreamzero0/dreamzero.

DreamZero server launch:
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \\
        --standalone --nproc_per_node=2 \\
        socket_test_optimized_AR.py --port 8000 --enable-dit-cache \\
        --model-path <merged_checkpoint_dir>

NOTE: No --lora-path flag. Merge LoRA offline before serving
(see dream-policy/merge_lora.py).

IMPORTANT: The repo dreamzero0/dreamzero is gated. Protocol details
were sourced from DeepWiki automated docs. Validate event keys/shapes
against the actual server once access is granted.
"""

import logging
import math
from typing import List, Optional

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import websocket  # websocket-client library

m.patch()  # monkey-patch msgpack to handle numpy arrays

logger = logging.getLogger(__name__)

# DreamZero inference server constants
DREAMZERO_IMG_H = 176
DREAMZERO_IMG_W = 320
DREAMZERO_ACTION_DIM = 7        # LIBERO: [dx, dy, dz, droll, dpitch, dyaw, gripper]
DREAMZERO_DEFAULT_PORT = 8000


def _resize_image(img_hwc: np.ndarray) -> np.ndarray:
    """Resize (H, W, 3) uint8 image to DreamZero resolution (176, 320, 3)."""
    return cv2.resize(img_hwc, (DREAMZERO_IMG_W, DREAMZERO_IMG_H),
                      interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] → axis-angle (3,)."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(max(1.0 - w * w, 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(w) / den).astype(np.float64)


def _build_joint_position(obs: dict) -> np.ndarray:
    """Build DreamZero joint_position (7,) from LIBERO observation.

    Maps: [eef_pos(3), eef_axisangle(3), gripper_mean(1)] → (7,)

    This is a zero-shot proxy mapping since LIBERO uses EEF-delta control
    and DreamZero was trained on AgiBot joint-space data. Results may vary.
    """
    eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)        # (3,)
    eef_aa  = _quat_to_axisangle(np.array(obs["robot0_eef_quat"]))     # (3,)
    gripper_qpos = np.array(obs["robot0_gripper_qpos"], dtype=np.float64)
    gripper_mean = np.array([gripper_qpos.mean()], dtype=np.float64)   # (1,)
    return np.concatenate([eef_pos, eef_aa, gripper_mean])              # (7,)


def _build_gripper_position(obs: dict) -> np.ndarray:
    """Build DreamZero gripper_position (1,) from LIBERO observation."""
    gripper_qpos = np.array(obs["robot0_gripper_qpos"], dtype=np.float64)
    return gripper_qpos[:1]  # (1,)


def _combine_action(joint_pos: np.ndarray, gripper_pos: np.ndarray) -> np.ndarray:
    """Combine (7,) joint_position and (1,) gripper_position → 7-DoF LIBERO action.

    DreamZero returns joint_position (7,) + gripper_position (1,) separately.
    LIBERO expects [dx, dy, dz, droll, dpitch, dyaw, gripper] (7,).
    Zero-shot mapping: use first 6 dims of joint_position as 6-DoF EEF delta,
    then append gripper_pos[0].
    """
    return np.concatenate([joint_pos[:6], gripper_pos[:1]]).astype(np.float32)


class DreamZeroClient:
    """WebSocket client for DreamZero WAM inference server.

    Args:
        host: Server hostname (default "localhost")
        port: Server port (default 8000)
        replan_steps: Steps to execute per action chunk before re-querying server
    """

    def __init__(self, host: str = "localhost", port: int = DREAMZERO_DEFAULT_PORT,
                 replan_steps: int = 24):
        self._url = f"ws://{host}:{port}"
        self.replan_steps = replan_steps
        self._language: str = ""
        self._action_queue: List[np.ndarray] = []
        self._ws: Optional[websocket.WebSocket] = None
        self._session_id: str = "libero_eval"

    def connect(self) -> None:
        logger.info("Connecting to DreamZero server at %s", self._url)
        self._ws = websocket.WebSocket()
        self._ws.connect(self._url, timeout=30)
        logger.info("Connected to DreamZero server")

    def disconnect(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def reset(self, language: str = "") -> None:
        """Send reset signal and clear action queue. Call at episode start."""
        self._language = language
        self._action_queue.clear()
        if self._ws is not None:
            payload = {"endpoint": "reset", "session_id": self._session_id}
            self._ws.send_binary(msgpack.packb(payload))

    def get_action(self, obs: dict,
                   agentview_img: np.ndarray,
                   wrist_img: Optional[np.ndarray] = None) -> np.ndarray:
        """Get next 7-DoF LIBERO action for the current observation.

        Args:
            obs: Raw LIBERO observation dict (for proprioception)
            agentview_img: (H, W, 3) uint8 — already 180°-flipped by get_safelibero_image
            wrist_img: (H, W, 3) uint8 — wrist camera; falls back to agentview if None

        Returns:
            np.ndarray (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        if self._action_queue:
            return self._action_queue.pop(0)

        actions = self._query_server(obs, agentview_img, wrist_img)
        self._action_queue = list(actions[1:])
        return actions[0]

    def _query_server(self, obs: dict, agentview_img: np.ndarray,
                      wrist_img: Optional[np.ndarray]) -> List[np.ndarray]:
        """Send observation to server, return decoded action chunk."""
        ext0 = _resize_image(agentview_img)
        ext1 = _resize_image(agentview_img)  # duplicate — no second external camera in LIBERO
        wrist = _resize_image(wrist_img if wrist_img is not None else agentview_img)

        joint_pos   = _build_joint_position(obs)
        gripper_pos = _build_gripper_position(obs)

        payload = {
            "endpoint": "infer",
            "observation/exterior_image_0_left": ext0,
            "observation/exterior_image_1_left": ext1,
            "observation/wrist_image_left":      wrist,
            "observation/joint_position":         joint_pos,
            "observation/gripper_position":       gripper_pos,
            "prompt":     self._language,
            "session_id": self._session_id,
        }

        self._ws.send_binary(msgpack.packb(payload))
        raw = self._ws.recv()
        response = msgpack.unpackb(raw if isinstance(raw, bytes) else raw.encode())

        joint_actions   = np.array(response["action.joint_position"],   dtype=np.float32)  # (N, 7)
        gripper_actions = np.array(response["action.gripper_position"], dtype=np.float32)  # (N, 1)

        return [
            _combine_action(joint_actions[i], gripper_actions[i])
            for i in range(len(joint_actions))
        ]
