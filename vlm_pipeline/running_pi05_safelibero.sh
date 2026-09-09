# Step 1: allocate the node
salloc -J aegis_job -p GPU --gres=gpu:h100-80:1 --time=08:00:00

# Step 2: note the node name that appears in your prompt, e.g. v001
# It will look like: (aegis) [asingal@v001 ~]$

# Step 3: activate env and start server
conda activate aegis
cd ~/vlsa-aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONPATH=$PYTHONPATH:$(pwd)/openpi/src

python openpi/scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=$(pwd)/checkpoints/pi05_libero

# Leave this terminal running — wait until you see:
# INFO:websockets.server:server listening on 0.0.0.0:8000


conda activate libero_env
cd /ocean/projects/cis250185p/asingal/vlsa-aegis
export PYTHONPATH=$PYTHONPATH:$(pwd)/../SafeLIBERO/safelibero
export PYTHONPATH=$PYTHONPATH:$(pwd)/main
export PYTHONPATH=$PYTHONPATH:$(pwd)/openpi/src

#find /ocean/projects/cis250185p/asingal/envs/libero -name "libstdc++.so.6" 2>/dev/null
#copy the above to the below command
#export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libero/lib/libstdc++.so.6

python main/pi05_evaluation.py \
    --task-suite-name safelibero_spatial \
    --safety-level I \
    --task-index 0 \
    --episode-index 0 1 2 3 4 5 \
    --video-out-path ../rollouts/pi0_5