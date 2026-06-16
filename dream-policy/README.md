# DreamZero WAM Integration

DreamZero-LIBERO-LoRA (community fine-tune on LIBERO-90) evaluated on SafeLIBERO.

## Setup
1. `bash setup_dreamzero_env.sh` — create conda env (Python 3.11 + PyTorch 2.8+cu129)
2. `bash download_checkpoint.sh` — download base model (~56GB) + LoRA adapter (217MB)
3. `bash start_dreamzero_server.sh` — launch inference server (2× GPU)

## Evaluation
See `vlm_pipeline/run_safelibero_dreamzero_eval.py` and `slurm/eval_dreamzero_safelibero_spatial_L1.slurm`.
