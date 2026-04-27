python -m vlm_prompt_runner.run_prompt_experiment --prompts-dir prompts/obstacle_id --prompts p3_candidate_list --model qwen3-vl-8b --suite safelibero_spatial --level I --task 0 --results-out vlm_prompt_runner/results/test.json

HF_HOME=/ocean/projects/cis250185p/asingal/hf_cache \
  python -m vlm_prompt_runner.run_prompt_experiment \
      --prompts-dir prompts/obstacle_id \
      --prompts p3_candidate_list \
      --model qwen3-vl-8b \
      --suite safelibero_spatial \
      --level I --task 0 \
      --results-out vlm_prompt_runner/results/test.json

python -m vlm_prompt_runner.run_prompt_experiment \     
        --prompts-dir prompts/obstacle_id \                                                                                                                           
        --model qwen3-vl-8b \
        --suite safelibero_spatial \                                                                                                                                  
        --level I --task 0 \                                
        --output-base vlm_prompt_runner/outputs/qwen3_vl_8b \
        --results-out vlm_prompt_runner/results/phase1_qwen3_vl_8b.json 

HF_HOME=/ocean/projects/cis250185p/asingal/hf_cache python -m vlm_prompt_runner.run_prompt_experiment --prompts-dir prompts/obstacle_id --model qwen3-vl-8b --suite safelibero_spatial --level I --task 0 --output-base vlm_prompt_runner/outputs/qwen3_vl_8b --results-out vlm_prompt_runner/results/phase1_qwen3_vl_8b.json
HF_HOME=/ocean/projects/cis250185p/asingal/hf_cache python -m vlm_prompt_runner.run_prompt_experiment --prompts-dir prompts/obstacle_id --model qwen2.5-vl-7b --suite safelibero_spatial --level I --task 0 --output-base vlm_prompt_runner/outputs/qwen25_vl_7b --results-out vlm_prompt_runner/results/phase1_qwen25_vl_7b.json
HF_HOME=/ocean/projects/cis250185p/asingal/hf_cache python -m vlm_prompt_runner.run_prompt_experiment --prompts-dir prompts/obstacle_id --model qwen2.5-vl-3b --suite safelibero_spatial --level I --task 0 --output-base vlm_prompt_runner/outputs/qwen25_vl_3b --results-out vlm_prompt_runner/results/phase1_qwen25_vl_3b.json