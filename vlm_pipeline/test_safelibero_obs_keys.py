"""Quick test to check SafeLIBERO observation keys"""
import sys
import os

# Add SafeLIBERO to path
sys.path.insert(0, "SafeLIBERO/safelibero")

from libero.libero import benchmark, get_libero_path

# Create environment
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["safelibero_spatial"](safety_level="I")
task = task_suite.get_task(0)

task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
from libero.libero.envs import OffScreenRenderEnv

env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=512,
    camera_widths=512
)

# Reset and get observation
obs = env.reset()

print("=" * 60)
print("SafeLIBERO Observation Keys:")
print("=" * 60)
for key in sorted(obs.keys()):
    if "image" in key.lower() or "depth" in key.lower() or "seg" in key.lower():
        val = obs[key]
        if hasattr(val, 'shape'):
            print(f"  {key:40s} -> shape: {val.shape}, dtype: {val.dtype}")
        else:
            print(f"  {key:40s} -> {type(val)}")

env.close()
