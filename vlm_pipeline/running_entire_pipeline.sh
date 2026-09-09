Build ellipsoids once per task:                                                       
python3 build_cbf_ellipsoids.py --vlm_json results/m1_task0_ep00.json --metadata vlm_inputs/safelibero_spatial/level_I/task_0/episode_00/metadata.json --output cbf_outputs/exclude_target_from_cbf/task_0_ellipsoids.json

# Run evaluation with CBF (requires GPU):                                               
conda activate openvla_libero_merged                                                  
export MUJOCO_GL=egl                                                                  
python3 run_libero_eval_with_cbf.py --task_suite_name=safelibero_spatial --num_trials_per_task=1
                                                                                        
  Outputs:                                                     
  - rollouts/{DATE}/...mp4 - Episode videos (same as original)                          
  - experiments/logs/trajectories/.../task_0/episode_00.npz - CBF trajectory data (NEW) 
                                                                                       
  The integration is complete and ready for GPU testing!  
