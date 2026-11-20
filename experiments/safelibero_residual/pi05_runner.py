"""pi0.5 as the base policy, behind the same interface as controllers.BaseRunner.

The chunk comes from an openpi websocket server (vlsa-aegis/openpi serve_policy, config pi05_libero) with
exactly the observation the reference driver sends (vlm_pipeline/run_safelibero_pi05_eval.py): both frames
flipped 180 deg and resize-with-pad to 224, state = eef pos + axis-angle + gripper qpos, prompt = task language.
The env renders at 256 px (the driver's resolution); the flow-side models (frozen flow-base features and the
residual's own encoders) see the same frames area-downsampled to 128 px, in raw env orientation.

Normalization: the pi0.5 chunk A (H=10, env units) is mapped into the flow base's [-1, 1] action space,
x = 2 (A - amin) / (amax - amin) - 1, without clipping, so unnorm(x) returns A exactly (the gripper is not
re-binarized). The residual therefore sees base chunks on the same scale as in the flow-base track.
"""
import math

import numpy as np
import torch

from controllers import BaseRunner

FLOW_RES = 128
PI_RES = 224


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def down(img, res=FLOW_RES):
    import cv2
    return img if img.shape[0] == res else cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)


class Pi05Runner(BaseRunner):
    """flow_ckpt: the suite's flow base, used only for its frozen features and action range."""

    H = 10

    def __init__(self, flow_ckpt, device="cuda", flow_steps=10, host="localhost", port=8000):
        super().__init__(flow_ckpt, device, flow_steps)
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy as wcp
        self.image_tools = image_tools
        self.client = wcp.WebsocketClientPolicy(host, port)
        self.uri = f"ws://{host}:{port}"
        self._connect()
        self.lang = None

    def _connect(self):
        """Replace the client's socket with one without keepalive pings. The server's event loop blocks during
        inference (and for minutes while JAX compiles the first call, with several workers queued), so the default
        20 s ping timeout closed every connection on 2026-11-02 (jobs 56300-56307, all episodes errored)."""
        import websockets.sync.client
        try:
            self.client._ws.close()
        except Exception:
            pass
        self.client._ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None,
                                                         ping_interval=None, open_timeout=600)
        self.client._ws.recv()  # server metadata

    def task_id(self, lang):
        self.lang = lang
        return 0  # features use the flow base's task embedding below

    def obs_tensors(self, obs, tid):
        import sl_env
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(self.dev)  # noqa: E731
        flow_tid = self.meta["langs"].index(self.lang)
        return (T(down(obs["agentview_image"])), T(down(obs["robot0_eye_in_hand_image"])),
                T((sl_env.proprio(obs) - self.pm) / self.ps), torch.tensor([flow_tid], device=self.dev))

    def query(self, obs):
        it = self.image_tools
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        el = {"observation/image": it.convert_to_uint8(it.resize_with_pad(img, PI_RES, PI_RES)),
              "observation/wrist_image": it.convert_to_uint8(it.resize_with_pad(wrist, PI_RES, PI_RES)),
              "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                   obs["robot0_gripper_qpos"])),
              "prompt": str(self.lang)}
        from websockets.exceptions import ConnectionClosed
        try:
            out = self.client.infer(el)
        except ConnectionClosed:  # one reconnect, then let the episode error out as before
            self._connect()
            out = self.client.infer(el)
        return np.asarray(out["actions"], np.float32)[: self.H]

    def chunk(self, obs, tid):
        """(normalized pi0.5 chunk (H,7) tensor, frozen flow-base features (1,F))."""
        A = self.query(obs)
        x = 2 * (A - self.amin) / (self.amax - self.amin) - 1
        with torch.no_grad():
            f = self.model.features(*self.obs_tensors(obs, tid), aug=False)
        return torch.from_numpy(x).to(self.dev), f

    def unnorm(self, x):
        return (np.asarray(x, np.float32) + 1) / 2 * (self.amax - self.amin) + self.amin
