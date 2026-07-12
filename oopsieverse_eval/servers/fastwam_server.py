"""Minimal REST server wrapping the off-the-shelf Fast-WAM checkpoint
(yuanty/fastwam, libero_uncond_2cam224 -- 2-camera, 224x224, LIBERO-trained).

Run inside the `oopsieverse_fastwam_srv` conda env:

    conda activate oopsieverse_fastwam_srv
    export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
    export DIFFSYNTH_MODEL_BASE_PATH=<repo>/fast-wam/checkpoints
    export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
    python oopsieverse_eval/servers/fastwam_server.py \
        --ckpt_path <repo>/fast-wam/checkpoints/libero_uncond_2cam224.pt \
        --stats_path <repo>/fast-wam/checkpoints/libero_uncond_2cam224_dataset_stats.json \
        --port 8300

Model loading mirrors vlm_pipeline/run_safelibero_fastwam_eval.py's
`load_fastwam_model()` (Hydra config compose against the FastWAM package's
own `sim_libero` config -- a checkpoint-architecture config, not anything
SafeLIBERO-specific) -- reused as-is since it's the model's own loading
contract, not benchmark eval code.

POST /act with a json-numpy-encoded body:
    {"primary_image": <H,W,3 uint8>, "wrist_image": <H,W,3 uint8>,
     "proprio": <8,> float32 [eef_pos(3), eef_axisangle(3), gripper_qpos(2)],
     "instruction": "pick up the egg", "action_horizon": 32,
     "num_inference_steps": 20}
returns a json-numpy-encoded [T, 7] action-chunk array, already
denormalized to physical units AND gripper-flipped to robosuite convention
(-1=open, +1=close) -- callers just slice off `replan_steps` and feed
directly into `build_env_action`.

NOTE on image orientation: this server does NOT apply Fast-WAM's own
180-degree LIBERO-specific image flip (see run_robocasa_pi05_eval.py's
docstring for the same judgment call and rationale) -- OopsieVerse's
RoboCasaAdapter already normalizes orientation internally.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import json_numpy

json_numpy.patch()

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastwam_server")

# `fast-wam/FastWAM` is gitignored in the main repo (large local checkout,
# not tracked git content) -- it does not exist inside this or any worktree,
# only in the main repo's working directory. Reference that fixed absolute
# path directly (read-only) rather than deriving from this file's location.
_HERE = Path(__file__).resolve().parent
_MAIN_REPO = Path("/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents")
_FASTWAM_REPO = _MAIN_REPO / "fast-wam" / "FastWAM"
_FASTWAM_SRC = _FASTWAM_REPO / "src"
_FASTWAM_CONFIGS = _FASTWAM_REPO / "configs"
for _p in (str(_FASTWAM_SRC),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def center_crop_resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    target_ratio = width / height
    cur_ratio = w / h
    if cur_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        img = img[:, x0 : x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        img = img[y0 : y0 + new_h, :]
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


class FastWAMServer:
    def __init__(self, ckpt_path: str, stats_path: str, gpu_id: int = 0):
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        self.device = f"cuda:{gpu_id}"
        self.dtype = torch.bfloat16

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(_FASTWAM_CONFIGS), version_base="1.3"):
            cfg = compose(
                config_name="sim_libero",
                overrides=[
                    f"ckpt={ckpt_path}",
                    f"gpu_id={gpu_id}",
                    "EVALUATION.rand_device=cpu",
                    "EVALUATION.text_cfg_scale=1.0",
                    "EVALUATION.binarize_gripper=false",
                    "model.load_text_encoder=true",
                    "model.skip_dit_load_from_pretrain=true",
                    "model.action_dit_pretrained_path=null",
                ],
            )
        self.cfg = cfg

        logger.info("Instantiating Fast-WAM model on %s...", self.device)
        self.model = instantiate(cfg.model, model_dtype=self.dtype, device=self.device)
        logger.info("Loading checkpoint: %s", ckpt_path)
        self.model.load_checkpoint(ckpt_path)
        self.model = self.model.to(self.device).eval()

        self.processor = instantiate(cfg.data.train.processor).eval()
        dataset_stats = load_dataset_stats_from_json(str(stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        video_size = cfg.data.train.get("video_size", [224, 448])
        self.input_h = int(video_size[0])
        self.input_w = int(video_size[1])
        self.num_cameras = int(self.processor.num_output_cameras)
        logger.info(
            "Model ready. video_size=%s num_cameras=%d action_dim=%d proprio_dim=%d",
            list(video_size), self.num_cameras,
            self.processor.action_output_dim, self.processor.proprio_output_dim,
        )

    def _build_image_tensor(self, primary_image: np.ndarray, wrist_image: np.ndarray) -> torch.Tensor:
        image_meta = self.processor.shape_meta["images"]
        concat_mode = self.cfg.data.train.get("concat_multi_camera", "horizontal")

        def meta_hw(idx):
            shape = image_meta[idx]["shape"]
            return int(shape[1]), int(shape[2])

        if self.num_cameras == 1:
            ph, pw = meta_hw(0)
            rgb = center_crop_resize(primary_image, width=pw, height=ph)
        elif self.num_cameras == 2:
            ph, pw = meta_hw(0)
            wh, ww = meta_hw(1)
            primary_resized = center_crop_resize(primary_image, width=pw, height=ph)
            wrist_resized = center_crop_resize(wrist_image, width=ww, height=wh)
            axis = 1 if concat_mode == "horizontal" else 0
            rgb = np.concatenate([primary_resized, wrist_resized], axis=axis)
        else:
            raise ValueError(f"Unsupported num_output_cameras={self.num_cameras}")

        img_tensor = (
            torch.tensor(rgb, dtype=self.dtype).permute(2, 0, 1).unsqueeze(0).to(device=self.device)
        )
        return img_tensor * (2.0 / 255.0) - 1.0

    def _normalize_proprio(self, proprio: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        state_key = state_meta[0]["key"]
        state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_meta = self.processor.shape_meta["action"]
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        action = action.to(dtype=torch.float32, device="cpu")
        return normalizer.backward(action).numpy()

    def predict(
        self, primary_image, wrist_image, proprio, instruction: str,
        action_horizon: int = 32, num_inference_steps: int = 20,
    ) -> np.ndarray:
        img_tensor = self._build_image_tensor(np.asarray(primary_image), np.asarray(wrist_image))
        proprio_tensor = self._normalize_proprio(np.asarray(proprio, dtype=np.float32))

        with torch.no_grad():
            pred = self.model.infer_action(
                prompt=DEFAULT_PROMPT.format(task=instruction),
                input_image=img_tensor,
                action_horizon=action_horizon,
                proprio=proprio_tensor,
                num_inference_steps=num_inference_steps,
                negative_prompt="",
                text_cfg_scale=1.0,
                rand_device="cpu",
            )

        action_chunk_norm = pred["action"]  # [T, D], normalized
        action_chunk = self._denormalize_action(action_chunk_norm)[0]  # [T, D]

        # Gripper convention: undo [0,1] encoding, then flip sign for robosuite.
        action_chunk[..., -1] = action_chunk[..., -1] * 2.0 - 1.0
        action_chunk[..., -1] = action_chunk[..., -1] * -1.0
        return action_chunk.astype(np.float32)


def build_app(server: FastWAMServer) -> FastAPI:
    app = FastAPI()

    @app.post("/act")
    async def act(request: Request):
        payload = json_numpy.loads(await request.body())
        try:
            action_chunk = server.predict(
                primary_image=payload["primary_image"],
                wrist_image=payload["wrist_image"],
                proprio=payload["proprio"],
                instruction=payload["instruction"],
                action_horizon=payload.get("action_horizon", 32),
                num_inference_steps=payload.get("num_inference_steps", 20),
            )
        except Exception:
            logger.exception("Inference failed")
            return JSONResponse(status_code=500, content={"error": "inference_failed"})
        return JSONResponse(content=json_numpy.dumps(action_chunk))

    @app.get("/health")
    async def health():
        return {"status": "ok", "num_cameras": server.num_cameras}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        default=str(_MAIN_REPO / "fast-wam" / "checkpoints" / "libero_uncond_2cam224.pt"),
    )
    parser.add_argument(
        "--stats_path",
        default=str(
            _MAIN_REPO / "fast-wam" / "checkpoints" / "libero_uncond_2cam224_dataset_stats.json"
        ),
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    server = FastWAMServer(ckpt_path=args.ckpt_path, stats_path=args.stats_path, gpu_id=args.gpu_id)
    app = build_app(server)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
