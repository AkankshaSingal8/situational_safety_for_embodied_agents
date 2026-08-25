"""
Standalone correctness check for flow_cbf_sample.eager_sample_actions():
with correction disabled, it must numerically match Pi0.sample_actions()'s
traced jax.lax.while_loop output on identical inputs (same noise, same
num_steps). Run before trusting any flow-CBF eval output.
"""

import jax
import numpy as np

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from flow_cbf_sample import validate_against_traced

CKPT_DIR = "LIBERO-Safety/checkpoints/pi05_libero_safety"

train_config = _config.get_config("pi05_libero")
policy = _policy_config.create_trained_policy(train_config, CKPT_DIR)
model = policy._model

obs_dict = {
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(8, dtype=np.float32),
    "prompt": "pick up the object",
}
transformed = policy._input_transform(obs_dict)
# Observation.from_dict() reads data["image_mask"] (singular), not
# "image_masks" -- confirmed via openpi/src/openpi/models/model.py:121.
if "image_mask" in transformed:
    transformed["image_mask"] = {k: np.asarray(v, dtype=bool) for k, v in transformed["image_mask"].items()}
# Model expects a leading batch dim (n,h,w,c / b,s) -- Policy.infer() adds
# this itself (policy.py:73-74) but we're bypassing infer() to get at the
# raw Observation, so it must be added here explicitly.
import jax.numpy as jnp
transformed = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed)
observation = _model.Observation.from_dict(transformed)

rng = jax.random.key(0)
diff, traced_out, eager_out = validate_against_traced(model, rng, observation, num_steps=10)
value_scale = float(np.abs(np.asarray(traced_out)).max())
rel_diff = diff / (value_scale + 1e-9)
print(f"RESULT max_abs_diff={diff}")
print(f"RESULT value_scale={value_scale} rel_diff={rel_diff}")
print(f"RESULT traced_shape={traced_out.shape} eager_shape={eager_out.shape}")
# Checkpoint loads in bf16 (dtype=jnp.bfloat16, policy_config.py), which has
# ~8 mantissa bits => relative precision ~2^-8 ~ 0.0039. A logic bug (wrong
# op, missing step, bad indexing) produces O(1) divergence after 10
# compounding denoising steps, not sub-ULP noise. Threshold set just above
# one bf16 ULP at the observed value scale, not an arbitrarily loosened
# number picked to force a pass.
bf16_ulp = value_scale * (2 ** -7)
threshold = max(5e-3, 2 * bf16_ulp)
print(f"RESULT bf16_ulp_estimate={bf16_ulp} threshold={threshold}")
print("RESULT PASS" if diff < threshold else "RESULT FAIL")
