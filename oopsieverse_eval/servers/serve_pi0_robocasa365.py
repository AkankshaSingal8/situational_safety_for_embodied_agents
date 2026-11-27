"""Serve the official RoboCasa365 pi0 checkpoint with our openpi checkout.

Checkpoint: HF robocasa/robocasa365_checkpoints, pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000
(robocasa-benchmark leaderboard pi0 entry). Trained with robocasa-benchmark/openpi@43fd9da config
"pi0_robocasa_pretrain_human300": pi0_base init, Pi0Config(max_token_len=96), RobocasaInputs/Outputs.
Our openpi has no RoboCasa config, so the fork's transforms are reproduced here verbatim
(src/openpi/policies/robocasa_policy.py) and norm stats are read from the checkpoint's
assets/norm_stats.json (the fork saves them flat, not under assets/<asset_id>/).

    python serve_pi0_robocasa365.py --ckpt <dir> --port 8000     (run with openpi's env, cwd = openpi)
"""
import argparse
import dataclasses
import logging
import pathlib

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0
        base = _parse_image(data["observation/image"])
        wrist = _parse_image(data["observation/wrist_image"])
        # robocasa-benchmark/openpi@ca4c6d7: pi0.5 uses all 3 cameras (agentview_right in the
        # right_wrist slot); pi0 pads that slot with zeros and masks it out.
        if self.model_type == _model.ModelType.PI05:
            right, right_mask = _parse_image(data["observation/right_image"]), np.True_
        else:
            right, right_mask = np.zeros_like(base), (np.False_ if mask_padding else np.True_)
        inputs = {
            "state": transforms.pad_to_dim(data["observation/state"], self.action_dim),
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": right},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": right_mask},
        }
        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :12])}


@dataclasses.dataclass(frozen=True)
class RobocasaDataConfig(_config.DataConfigFactory):
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(),
            data_transforms=transforms.Group(
                inputs=[RobocasaInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
                outputs=[RobocasaOutputs()]),
            model_transforms=_config.ModelTransformFactory()(model_config),
            # The fork never enables quantile norm (DataConfig.use_quantile_norm stays False, and the
            # checkpoints' norm_stats.json has only mean/std); our openpi turns it on for pi0.5.
            use_quantile_norm=False,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pi05", action="store_true", help="pi05_pretrain_human300: Pi0Config(pi05=True, max_token_len=200)")
    a = ap.parse_args()
    model = (pi0_config.Pi0Config(pi05=True, max_token_len=200) if a.pi05
             else pi0_config.Pi0Config(max_token_len=96))
    cfg = _config.TrainConfig(name="pi05_pretrain_human300" if a.pi05 else "pi0_robocasa_pretrain_human300",
                              model=model, data=RobocasaDataConfig(repo_id="robocasa365"))
    ns = _normalize.deserialize_json((pathlib.Path(a.ckpt) / "assets" / "norm_stats.json").read_text())
    policy = policy_config.create_trained_policy(cfg, pathlib.Path(a.ckpt), norm_stats=ns)
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=a.port,
                                                           metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
