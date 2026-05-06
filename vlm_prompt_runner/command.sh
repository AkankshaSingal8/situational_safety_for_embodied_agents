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
HF_HOME=/ocean/projects/cis250185p/asingal/hf_cache python -m vlm_prompt_runner.run_prompt_experiment --prompts-dir prompts/obstacle_id --prompts p6_visual_path p7_per_object_binary --model qwen3-vl-8b --suite safelibero_spatial --level I --task 0 --output-base vlm_prompt_runner/outputs/qwen3_vl_8b --results-out vlm_prompt_runner/results/phase2_qwen3_vl_8b.json
HF_HOME=/ocean/projects/cis250185p/asingal/hf_cache python -m vlm_prompt_runner.run_prompt_experiment --prompts-dir prompts/obstacle_id --prompts p6_visual_path p8_ranked_collision --model qwen3-vl-8b --suite safelibero_spatial --level I --task 0 --output-base vlm_prompt_runner/outputs/qwen3_vl_8b --results-out vlm_prompt_runner/results/phase3_qwen3_vl_8b.json


HF_HOME=/ocean/projects/cis250185p/asingal/hf_cache python -m vlm_prompt_runner.run_prompt_experiment --prompts-dir prompts/obstacle_id --model qwen2.5-vl-7b --suite safelibero_spatial --level I --task 0 --output-base vlm_prompt_runner/outputs/qwen25_vl_7b --results-out vlm_prompt_runner/results/phase1_qwen25_vl_7b.json
HF_HOME=/ocean/projects/cis250185p/asingal/hf_cache python -m vlm_prompt_runner.run_prompt_experiment --prompts-dir prompts/obstacle_id --model qwen2.5-vl-3b --suite safelibero_spatial --level I --task 0 --output-base vlm_prompt_runner/outputs/qwen25_vl_3b --results-out vlm_prompt_runner/results/phase1_qwen25_vl_3b.json

#api command
python -m vlm_prompt_runner.run_multi_model_experiment --prompts-dir prompts/obstacle_id --prompts p8_ranked_collision --models gpt-5.5 --suite safelibero_spatial --level I --task 0 --results-out vlm_prompt_runner/results/test_gpt55_p8.json

python -m vlm_prompt_runner.run_experiment --prompts-dir prompts/obstacle_id --prompts p8_ranked_collision --models gpt-5.5 --suite safelibero_spatial --level I --task 0 --results-out vlm_prompt_runner/results/test_gpt55_p8.json

python -m vlm_prompt_runner.run_experiment --prompts-dir prompts/obstacle_id --prompts p8_ranked_collision --models gpt-5.5 --suite  safelibero_spatial --level I --task 0 --results-out vlm_prompt_runner/results/test_gpt55_p8.json


rm -rf vlm_prompt_runner/outputs/gpt_5_5/ && 
python -m vlm_prompt_runner.run_experiment --prompts-dir prompts/obstacle_id --prompts p1_vision_only --models gpt-5.5 --suite safelibero_spatial --level I --task 0 --results-out vlm_prompt_runner/gpt_5_5/p1_vision_only.json


python -m vlm_prompt_runner.run_experiment --prompts-dir prompts/obstacle_id --prompts p8_ranked_collision --models gpt-5.5 --suite safelibero_spatial --level I --task 0 --results-out vlm_prompt_runner/results/test_gpt55_p8.json