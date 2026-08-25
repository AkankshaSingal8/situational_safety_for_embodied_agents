#!/bin/bash
#SBATCH --job-name=fol_v5_smoke
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=5
#SBATCH --partition=GPU-shared --gres=gpu:v100-32:1
#SBATCH --time=02:00:00 --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v5_smoke_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v5_smoke_%j.err

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs "$WT/fol_v5_smoke" "$WT/fol_v5_smoke_video"
echo "=== FOL v5 + Qwen VLM Smoke Test (safelibero_spatial L1 n=2) ==="
echo "Start: $(date)"

module load anaconda3
conda activate openvla_libero_merged

export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# ── Deploy v5 files ──
echo "Deploying v5 filter files..."
cp "$WT/v5/fol_safety_filter/filter.py"       "$WT/fol_safety_filter/filter.py"
cp "$WT/v5/fol_safety_filter/vlm_grounder.py" "$WT/fol_safety_filter/vlm_grounder.py"
cp "$WT/v5/fol_safety_filter/rule_composer.py" "$WT/fol_safety_filter/rule_composer.py"
echo "Deployed filter.py, vlm_grounder.py, rule_composer.py"

# ── Start Qwen server (7B 4-bit: ~8GB + OpenVLA ~16GB = ~24GB on V100-32) ──
QWEN_LOG="$WT/fol_v5_smoke/qwen_server.log"
conda run --no-capture-output -n qwen \
    python "$REPO/vlm_pipeline/qwen_vlm_server.py" \
    --port 5001 --model qwen2.5-vl-7b --load_in_4bit \
    > "$QWEN_LOG" 2>&1 &
QWEN_PID=$!
echo "Qwen server PID: $QWEN_PID (log: $QWEN_LOG)"

# Poll until ready (max 8 min)
echo "Waiting for Qwen server..."
READY=0
for i in $(seq 1 96); do
    if curl -sf http://localhost:5001/health > /dev/null 2>&1; then
        READY=1
        echo "Qwen server ready after ${i}x5s = $((i*5))s"
        break
    fi
    sleep 5
done

if [ "$READY" -eq 0 ]; then
    echo "ERROR: Qwen server did not start in 8 minutes. Aborting."
    kill "$QWEN_PID" 2>/dev/null
    exit 1
fi

# Verify /infer_fol_v5 endpoint exists
echo "Testing /infer_fol_v5 endpoint..."
curl -sf -X POST http://localhost:5001/infer_fol_v5 \
    -H "Content-Type: application/json" \
    -d '{"images_b64":[],"task_description":"pick up the bowl","candidates":["moka pot","bowl"],"parsed_target":"bowl"}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('v5 endpoint OK:', d)" \
    || echo "WARNING: /infer_fol_v5 endpoint test failed (proceeding anyway)"

export QWEN_SERVER_URL=http://localhost:5001
export PYTHONPATH="$WT/v5:$WT:$REPO/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}"

echo "Running FOL v5 smoke test..."
python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name safelibero_spatial --safety_level I \
    --num_trials_per_task 2 --use_fol_filter True --fol_level 1 \
    --results_output_dir "$WT/fol_v5_smoke" \
    --video_output_dir "$WT/fol_v5_smoke_video" \
    --center_crop True --seed 7 --run_id_note "fol-v5-smoke"
EVAL_EXIT=$?

# ── Cleanup ──
echo "Shutting down Qwen server (PID=$QWEN_PID)"
kill "$QWEN_PID" 2>/dev/null
wait "$QWEN_PID" 2>/dev/null

echo "=== Done: $(date) (eval exit=$EVAL_EXIT) ==="
exit $EVAL_EXIT
