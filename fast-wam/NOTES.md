# Fast-WAM Integration Notes

Inspected from: https://github.com/yuantianyuan01/FastWAM (depth-1 clone, 2026-06-15)
Source: `/tmp/fastwam_inspect/`

---

## 1. Repository Layout

```
FastWAM/
  configs/
  experiments/
    libero/
      eval_libero_single.py   # main eval entrypoint
      libero_utils.py         # obs preprocessing helpers
      action_ensembler.py
      run_libero_manager.py
  scripts/
    preprocess_action_dit_backbone.py
  src/fastwam/
    runtime.py                # create_fastwam(), run_inference(), run_training()
    models/wan22/
      fastwam.py              # FastWAM class with infer_action / infer_joint / infer
      fastwam_joint.py
      fastwam_idm.py
      action_dit.py
      wan22.py
    datasets/
    trainer.py
    utils/
  pyproject.toml
```

---

## 2. `infer_action` Signature

Defined in `src/fastwam/models/wan22/fastwam.py`, line 906:

```python
def infer_action(
    self,
    prompt: Optional[str],
    input_image: torch.Tensor,          # shape [1, 3, H, W] or [3, H, W]; pixel values in [-1, 1]
    action_horizon: int,                # number of future action steps to predict
    proprio: Optional[torch.Tensor] = None,    # shape [D] or [1, D], already normalized
    context: Optional[torch.Tensor] = None,    # pre-encoded text context (alternative to prompt)
    context_mask: Optional[torch.Tensor] = None,
    negative_prompt: Optional[str] = None,
    text_cfg_scale: float = 1.0,
    num_inference_steps: int = 20,
    sigma_shift: Optional[float] = None,
    seed: Optional[int] = None,
    rand_device: str = "cpu",
    tiled: bool = False,
) -> dict[str, Any]:
```

### Return value

```python
{"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
#   key:  "action"
#   shape: [T, D]  where T = action_horizon, D = action_expert.action_dim
#   dtype: float32, on CPU
#   NOTE: these are NORMALIZED action latents — must call _denormalize_action() before executing
```

### Usage in eval loop (`eval_libero_single.py`, line 418-419)

```python
pred = model.infer_action(**infer_kwargs)
action = pred["action"]  # [T, D]
```

`infer_kwargs` is built at lines 389-412 and always includes:
- `prompt`, `input_image`, `action_horizon`, `negative_prompt`, `text_cfg_scale`,
  `num_inference_steps`, `proprio`, `sigma_shift`, `seed`, `rand_device`, `tiled`
- Optionally `num_video_frames` if the model signature supports it (checked via `inspect.signature`)

---

## 3. Observation Preprocessing

### Image flip + extraction (`libero_utils.py`, lines 45-57)

```python
def get_libero_image(obs):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {"image": img, "wrist_image": wrist_img}
```

Both `agentview_image` and `robot0_eye_in_hand_image` are flipped 180 degrees (both axes reversed).

### Resize + crop (`eval_libero_single.py`, lines 155-164)

```python
def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)
```

Target size comes from `cfg.data.train.video_size = [H, W]`. Per-camera sizes come from
`processor.shape_meta["images"][i]["shape"]` (format: [C, H, W]).

### Multi-camera concatenation (`eval_libero_single.py`, lines 208-223)

- 1 camera: primary only, resized to `shape_meta.images[0].shape`
- 2 cameras: primary + wrist, concatenated along axis=1 (horizontal, default) or axis=0 (vertical)
  controlled by `cfg.data.train.concat_multi_camera`

### Pixel normalization

```python
x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
x = x * (2.0 / 255.0) - 1.0   # maps [0,255] -> [-1, 1]
```

### Proprioception construction (`eval_libero_single.py`, lines 244-256)

```python
def _extract_sim_state(obs: dict) -> np.ndarray:
    state = np.concatenate((
        obs["robot0_eef_pos"],           # (3,) float64
        quat2axisangle(obs["robot0_eef_quat"]),  # (3,) axis-angle from (x,y,z,w) quat
        obs["robot0_gripper_qpos"],      # (2,) float64
    )).astype(np.float32)
    return state   # shape (8,)
```

---

## 4. Proprio Normalization API

Defined in `eval_libero_single.py`, lines 167-181:

```python
def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_key = processor.shape_meta["state"][0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]
```

Steps:
1. Wrap numpy array in a `{"state": {state_key: tensor}}` dict (batch dim added with unsqueeze(0))
2. Call `processor.action_state_transform(state_batch)` — applies any state transforms
3. Call `processor.normalizer.forward(state_batch)` — normalizes in-place
4. Extract normalized tensor from `state_batch["state"][state_key]`

The processor is a `FastWAMProcessor` from
`fastwam.datasets.lerobot.processors.fastwam_processor`. Its normalizer is set at eval time via:

```python
dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))   # from dataset_stats.json
processor.set_normalizer_from_stats(dataset_stats)
```

`dataset_stats.json` is auto-located: first tries `EVALUATION.dataset_stats_path`, then walks up
from checkpoint path checking up to 4 parent dirs for `dataset_stats.json`.

---

## 5. Action Denormalization

```python
def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    action_key = processor.shape_meta["action"][0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    denorm = normalizer.backward(action)   # .backward() is the denorm call
    return denorm.numpy()
```

After denormalization, the gripper action is post-processed:
```python
action[..., -1] = action[..., -1] * 2 - 1   # undo dataset's [0,1] gripper encoding
action = invert_gripper_action(action)        # multiply by -1 (robosuite sign convention)
```

---

## 6. Checkpoint Loading

### At eval time (`eval_libero_single.py`, line 117-118)

```python
def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
```

`load_checkpoint` is a method on the `FastWAM` model class (defined in
`src/fastwam/models/wan22/fastwam.py`). It is NOT `torch.load` — the model handles its own
deserialization internally.

### At inference time (`runtime.py`, lines 390-396)

```python
checkpoint_path = inference_cfg.get("checkpoint_path")
if checkpoint_path:
    model.load_checkpoint(checkpoint_path)
model.eval()
```

### Model instantiation

Models are instantiated via Hydra's `instantiate(cfg.model, ...)` with one of:
- `create_fastwam(...)` for the standard Fast-WAM model
- `create_fastwam_joint(...)` for joint video+action prediction
- `create_fastwam_idm(...)` for inverse-dynamics variant

---

## 7. `preprocess_action_dit_backbone.py` Arguments

```
python scripts/preprocess_action_dit_backbone.py \
    --model-config  PATH   # (required) path to model yaml, e.g. configs/model/fastwam.yaml
    --output        PATH   # (required) output .pt path for preprocessed ActionDiT backbone
    --device        STR    # (default: "cpu") device for loading/preprocessing
    --dtype         STR    # (default: "float32") one of: float32, float16, bfloat16
    --apply-alpha-scaling STR  # (default: "true") apply sqrt(dv/da) alpha when last dim is resized
```

Output format is a `.pt` dict with keys:
- `"policy"`: metadata dict with `skip_prefixes`, `alpha_scaling`, `interpolation`
- `"backbone_state_dict"`: the extracted/interpolated backbone weights
- `"meta"`: ActionDiT arch params (`hidden_dim`, `ffn_dim`, `num_layers`, etc.)

---

## 8. Python / PyTorch / CUDA Requirements

From `pyproject.toml`:

```
requires-python = ">=3.10"
torch==2.7.1+cu128         # CUDA 12.8
torchvision==0.22.1+cu128  # CUDA 12.8
torchcodec==0.5
accelerate==1.12.0
transformers==4.49.0
hydra-core==1.3.2
omegaconf==2.3.0
einops==0.8.1
numpy==1.26.4
pillow==12.0.0
deepspeed==0.18.5
wandb==0.23.1
huggingface-hub==0.29.2
```

**DEVIATION**: Requires CUDA 12.8 (cu128). The current project's `openvla_libero_merged`
environment uses torch 2.2.0 (CUDA 11.8). Fast-WAM will need a separate conda environment.

---

## 9. Additional Notes / Deviations from Expected API

1. **No `num_video_frames` in `infer_action` by default**: The signature does NOT include
   `num_video_frames`. The eval script checks for it with `inspect.signature` and only passes it
   if present. This parameter appears to be an extension for custom model variants.

2. **`infer_action` vs `infer_joint`**: `infer_action` returns only `{"action": tensor}`.
   `infer_joint` (line 726) returns `{"video": list[PIL.Image], "action": tensor}` and is used
   when `EVALUATION.visualize_future_video=True`.

3. **`proprio` is optional**: If `proprio_dim=None` at model creation, proprio is disabled and
   passing `proprio` to `infer_action` will raise an error.

4. **Prompt template**: Task description is wrapped as
   `DEFAULT_PROMPT.format(task=task_description)` where `DEFAULT_PROMPT` is imported from
   `fastwam.datasets.lerobot.robot_video_dataset`.

5. **action_horizon default**: Defaults to `cfg.data.train.num_frames - 1` (i.e., number of video
   training frames minus the conditioning frame). Can be overridden via
   `EVALUATION.action_horizon`.

6. **LIBERO env resolution**: `LIBERO_ENV_RESOLUTION = 256` (constant in `libero_utils.py`). This
   is the environment rendering resolution, separate from the model input resolution defined by
   `cfg.data.train.video_size`.

7. **Gripper convention**: Dataset stores gripper as 0=close, 1=open. After denorm, the eval code
   does `* 2 - 1` then `* -1` to convert to robosuite's -1=open, +1=close.

8. **No `action_cfg_scale`** in `infer_action`: The full `infer()` method (video-only inference)
   takes `action_cfg_scale`, but `infer_action` does not — it only has `text_cfg_scale`.
