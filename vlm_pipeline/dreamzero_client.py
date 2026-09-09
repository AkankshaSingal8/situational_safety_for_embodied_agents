# vlm_pipeline/dreamzero_client.py
"""
Client for DreamZero WAM inference server.

Protocol: WebSocket + openpi_client msgpack_numpy (same as pi0.5).
Client uses WebsocketClientPolicy from eval_utils.policy_client
(included in dream-policy/dreamzero repo).

Server entry point: socket_test_optimized_AR.py

DreamZero server launch (2 GPUs):
    conda activate /ocean/projects/cis250185p/asingal/envs/dreamzero_env
    cd dream-policy/dreamzero
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \\
        socket_test_optimized_AR.py --port 8000 --enable-dit-cache \\
        --model-path <merged_checkpoint_dir>

Server configuration expected by this client:
    n_external_cameras = 2
    needs_wrist_camera = True
    action_space = "joint_position"
    image_resolution = (180, 320)

Action output: (N, 8) numpy array — 7 joint positions + 1 gripper.
LIBERO 7-DoF mapping: first 6 joints → EEF delta (zero-shot proxy), action[7] → gripper.
"""

import logging
import math
import sys
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DREAMZERO = _REPO_ROOT / "dream-policy" / "dreamzero"
if str(_DREAMZERO) not in sys.path:
    sys.path.insert(0, str(_DREAMZERO))

from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402

logger = logging.getLogger(__name__)

# DreamZero server config (from test_client_AR.py and policy_server.py)
DREAMZERO_IMG_H = 180
DREAMZERO_IMG_W = 320
DREAMZERO_DEFAULT_PORT = 8000
DREAMZERO_ACTION_OUT_DIM = 8  # 7 joints + 1 gripper
LIBERO_ACTION_DIM = 7          # [dx, dy, dz, droll, dpitch, dyaw, gripper]


def _resize_image(img_hwc: np.ndarray) -> np.ndarray:
    """Resize (H, W, 3) uint8 image to DreamZero resolution (180, 320)."""
    return cv2.resize(img_hwc, (DREAMZERO_IMG_W, DREAMZERO_IMG_H),
                      interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] → axis-angle (3,)."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(max(1.0 - w * w, 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(w) / den).astype(np.float32)


def _build_joint_position(obs: dict) -> np.ndarray:
    """Build DreamZero joint_position (7,) from LIBERO observation.

    Zero-shot proxy: [eef_pos(3), eef_axisangle(3), gripper_mean(1)] → (7,).
    """
    eef_pos  = np.array(obs["robot0_eef_pos"],    dtype=np.float32)       # (3,)
    eef_aa   = _quat_to_axisangle(np.array(obs["robot0_eef_quat"]))       # (3,)
    gripper  = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)
    return np.concatenate([eef_pos, eef_aa, gripper[:1]])                  # (7,)


def _build_cartesian_position(obs: dict) -> np.ndarray:
    """Build DreamZero cartesian_position (6,) from LIBERO observation.

    Format: [eef_pos(3), eef_axisangle(3)].
    """
    eef_pos = np.array(obs["robot0_eef_pos"],  dtype=np.float32)          # (3,)
    eef_aa  = _quat_to_axisangle(np.array(obs["robot0_eef_quat"]))        # (3,)
    return np.concatenate([eef_pos, eef_aa])                               # (6,)


def _build_gripper_position(obs: dict) -> np.ndarray:
    """Build DreamZero gripper_position (1,) from LIBERO observation."""
    return np.array(obs["robot0_gripper_qpos"][:1], dtype=np.float32)


def _dreamzero_to_libero_action(action_8d: np.ndarray) -> np.ndarray:
    """Map DreamZero (8,) action to LIBERO 7-DoF.

    DreamZero: [joint1..joint7, gripper] (8,)
    LIBERO:    [dx, dy, dz, droll, dpitch, dyaw, gripper] (7,)
    Zero-shot mapping: first 6 joint dims as EEF delta, action[7] as gripper.
    """
    return np.concatenate([action_8d[:6], action_8d[7:8]]).astype(np.float32)


class DreamZeroClient:
    """Client for DreamZero WAM inference server via WebSocket + openpi_client protocol.

    Wraps eval_utils.policy_client.WebsocketClientPolicy (from dreamzero0/dreamzero repo).
    The server handles temporal frame accumulation; this client sends one frame per call.

    Args:
        host: Server hostname (default "localhost")
        port: Server port (default 8000)
        replan_steps: Steps to execute per action chunk before re-querying server
    """

    def __init__(self, host: str = "localhost", port: int = DREAMZERO_DEFAULT_PORT,
                 replan_steps: int = 24):
        self._host = host
        self._port = port
        self.replan_steps = replan_steps
        self._language: str = ""
        self._session_id: str = str(uuid.uuid4())
        self._action_queue: List[np.ndarray] = []
        self._client: Optional[WebsocketClientPolicy] = None

    def connect(self) -> None:
        logger.info("Connecting to DreamZero server at %s:%d ...", self._host, self._port)
        self._client = WebsocketClientPolicy(host=self._host, port=self._port)
        metadata = self._client.get_server_metadata()
        logger.info("DreamZero server metadata: %s", metadata)

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.reset({})
            except Exception:
                pass
            self._client = None

    def reset(self, language: str = "") -> None:
        """Send reset to server and clear action queue. Call at each episode start."""
        self._language = language
        self._session_id = str(uuid.uuid4())
        self._action_queue.clear()
        if self._client is not None:
            try:
                self._client.reset({})
            except Exception as e:
                logger.warning("Reset call failed: %s", e)

    def get_action(self, obs: dict,
                   agentview_img: np.ndarray,
                   wrist_img: Optional[np.ndarray] = None) -> np.ndarray:
        """Get next 7-DoF LIBERO action for the current observation.

        Args:
            obs: Raw LIBERO observation dict (for proprioception keys)
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
        """Build observation dict, call server, return list of 7-DoF actions."""
        ext0 = _resize_image(agentview_img)
        ext1 = _resize_image(agentview_img)  # duplicate — no second external cam in LIBERO
        wrist = _resize_image(wrist_img if wrist_img is not None else agentview_img)

        server_obs = {
            "observation/exterior_image_0_left": ext0,             # (180, 320, 3) uint8
            "observation/exterior_image_1_left": ext1,             # (180, 320, 3) uint8
            "observation/wrist_image_left":      wrist,            # (180, 320, 3) uint8
            "observation/joint_position":         _build_joint_position(obs),      # (7,)
            "observation/cartesian_position":     _build_cartesian_position(obs),  # (6,)
            "observation/gripper_position":       _build_gripper_position(obs),    # (1,)
            "prompt":     self._language,
            "session_id": self._session_id,
        }

        raw_actions = self._client.infer(server_obs)  # (N, 8) numpy array
        return [_dreamzero_to_libero_action(raw_actions[i]) for i in range(len(raw_actions))]
