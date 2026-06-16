# DreamZero Integration Research Notes

## GitHub Repository
- Repo: https://github.com/dreamzero0/dreamzero
- Server entry point: `socket_test_optimized_AR.py`
- DeepWiki docs: https://deepwiki.com/dreamzero0/dreamzero
- Status: Repo exists but clone returned empty tree (likely private/restricted content). Protocol details sourced from DeepWiki automated documentation.

## Server Protocol
- Transport: **WebSocket** (ws:// or wss://) — NOT Flask-SocketIO despite both packages being listed in pyproject.toml
- Serialization: **msgpack-numpy** (binary, not JSON)
- Pattern: Synchronous request-response
- Compression: Disabled for performance
- Health check: HTTP GET `/healthz` → 200 "OK"

### Launch Command
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone \
  --nproc_per_node=2 socket_test_optimized_AR.py \
  --model-path /path/to/checkpoint \
  --enable-dit-cache \
  --port 5000
```
(Alternatively use `torchrun` instead of `python -m torch.distributed.run`.)

### Send Event / Request (Client → Server)
- Endpoint routing key: `"endpoint"` field in request dict, set to `"infer"` or `"reset"`
- Serialization: msgpack_numpy

**Observation dictionary keys:**
| Key | Shape | dtype |
|-----|-------|-------|
| `observation/exterior_image_0_left` | (H, W, 3) | uint8 |
| `observation/exterior_image_1_left` | (H, W, 3) | uint8 |
| `observation/wrist_image_left`      | (H, W, 3) | uint8 |
| `observation/joint_position`        | (7,)       | float64 |
| `observation/gripper_position`      | (1,)       | float64 |
| `prompt`                            | str        | — |
| `session_id`                        | str        | optional, for session tracking |
| `endpoint`                          | str        | required: `"infer"` or `"reset"` |

**Notes:**
- The `needs_wrist_camera` and `n_external_cameras` fields in `PolicyServerConfig` suggest the number of image keys is configurable (some deployments may omit wrist or use only one exterior camera).
- Image resolution is set via server-side `image_resolution` config.

### Receive Event / Response (Server → Client)
- Serialization: msgpack_numpy

**Action dictionary keys:**
| Key | Shape | dtype |
|-----|-------|-------|
| `action.joint_position`   | (N, 7) | float32 |
| `action.gripper_position` | (N, 1) or (N,) | float32 |

Where **N = `open_loop_horizon`** (model configuration parameter — number of future timesteps returned per inference call). This means the server returns a *chunk* of actions, not a single action.

## CLI Arguments
```
--model-path PATH       Path to pretrained model checkpoint dir (HuggingFace format). REQUIRED.
--port INT              WebSocket server port. Default: 8000.
--enable-dit-cache      Enable DiT layer caching for faster inference (recommended).
                        ~0.6s latency on GB200, ~3s on H100.
--timeout-seconds INT   Distributed operation timeout in seconds. Default: 50000.
--index INT             Index suffix for output directory naming. Default: 0.
--max-chunk-size INT    Override maximum chunk size for action predictions. Default: None.
```

- `--model-path` flag: **YES** — exact flag name is `--model-path`
- `--lora-path` flag: **NO** — not documented in the server CLI. LoRA weights must be **merged into the base model offline** before running the server. The training config supports `save_lora_only=true`, so a merge step (PEFT `merge_and_unload()`) is needed before serving.

## Dependencies (from pyproject.toml)
Key packages relevant to integration:
- Python 3.11–3.12 required
- `torch` 2.8.0 / `torchvision` 0.23.0 / `torchaudio` 2.8.0
- `transformers` 4.51.3
- `diffusers` 0.30.2
- `peft` 0.5.0 (for LoRA fine-tuning)
- `msgpack` + `msgpack-numpy` (binary serialization for WebSocket)
- `python-socketio` >=5.13.0 + `flask-socketio` (present in deps but WebSocket is primary transport)
- `pyzmq` (inter-process communication)
- `mujoco` (simulation)
- `numpy` 1.26.4
- `deepspeed`, `tensorrt`, `nvidia-modelopt` (inference optimization)
- `ray[default]` 2.47.1 (distributed coordination)
- `huggingface-hub`, `wandb`

## Connecting from the LIBERO Eval Client

To connect to the DreamZero server from `run_safelibero_fastwam_eval.py` (or similar):

```python
import asyncio
import websockets
import msgpack
import msgpack_numpy as m
m.patch()

async def query_dreamzero(ws_uri, obs_dict, prompt, session_id=None):
    request = {
        "endpoint": "infer",
        "prompt": prompt,
        "observation/exterior_image_0_left": obs_dict["agentview_image"],  # (H, W, 3) uint8
        "observation/joint_position": obs_dict["robot0_joint_pos"],         # (7,) float64
        "observation/gripper_position": obs_dict["robot0_gripper_qpos"][:1],
    }
    if session_id:
        request["session_id"] = session_id
    async with websockets.connect(ws_uri, max_size=None,
                                  ping_interval=60, ping_timeout=600) as ws:
        await ws.send(msgpack.packb(request, default=m.encode))
        raw = await ws.recv()
        response = msgpack.unpackb(raw, object_hook=m.decode)
    # response["action.joint_position"] shape: (N, 7)
    # response["action.gripper_position"] shape: (N, 1)
    return response
```

**Key adapter work needed:**
1. LIBERO obs keys → DreamZero obs keys (image resize to server's `image_resolution`)
2. DreamZero returns N-step action chunk; LIBERO eval loop expects single 7-DoF action per step
3. Need to handle gripper: LIBERO uses 1D gripper command; DreamZero returns separate `action.gripper_position`
4. `reset` endpoint should be called at episode start

## Notes / Surprises
1. **Not Flask-SocketIO at the transport level**: Despite `python-socketio` and `flask-socketio` being in `pyproject.toml`, the actual server uses raw WebSocket with `msgpack-numpy` serialization, not Socket.IO events. The event model is a routing key inside the msgpack payload (`"endpoint": "infer"`), not Socket.IO `emit`/`on`.
2. **Chunked actions (open-loop horizon)**: Server returns N future actions per call, not one. The client must buffer and replay them, or re-query each step. Need to decide policy for SafeLIBERO eval.
3. **No --lora-path**: LoRA fine-tuned weights must be merged offline into a single checkpoint directory before starting the server. No runtime LoRA loading.
4. **Multi-GPU required**: Nominal deployment uses 2 GPUs via `torchrun --nproc_per_node=2`. Single-GPU fallback may be possible with `--nproc_per_node=1` but is untested for this integration.
5. **No LIBERO-specific support upstream**: DreamZero README documents DROID simulation and RoboArena evals; LIBERO obs/action space mapping is entirely our adapter work.
6. **Python 3.11+ required**: DreamZero requires Python 3.11–3.12. Our current `openvla_libero_merged` env is Python 3.10. The DreamZero server must run in a separate environment/process and be called over WebSocket.
7. **Repo content inaccessible via git clone**: `dreamzero0/dreamzero` clones as an empty tree (content is gated). Protocol details are sourced from DeepWiki automated docs and may have minor inaccuracies — validate against actual `socket_test_optimized_AR.py` once access is granted.
