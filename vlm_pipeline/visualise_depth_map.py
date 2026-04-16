import numpy as np
import matplotlib.pyplot as plt

depth = np.load("vlm_inputs/safelibero_spatial/level_I/task_0/episode_00/agentview_depth.npy")
depth = np.flipud(depth)

valid = depth[depth > 0]
vmin = np.percentile(valid, 2)
vmax = np.percentile(valid, 98)

plt.figure(figsize=(6, 6))
plt.imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax)
plt.colorbar(label="Depth (meters)")
plt.title("Clipped depth map")
plt.axis("off")
plt.savefig("vlm_inputs/safelibero_spatial/level_I/task_0/episode_00/agentview_depth.png", bbox_inches="tight", pad_inches=0)
plt.show()