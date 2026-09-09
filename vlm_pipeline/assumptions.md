 Assumptions / Open Questions

 1. prepare_observation() compatibility: Uses get_libero_wrist_image() from openvla-oft. SafeLIBERO only enables agentview camera (no
 wrist). Need to verify whether get_libero_wrist_image(obs) raises or returns zeros when wrist camera is absent. If it raises,
 prepare_observation() must be overridden to use a zeroed wrist image or duplicate agentview.
   - Action required: Check safelibero_utils.py and run_libero_eval_with_cbf.py to see how they handle this — likely have a workaround
 already.
 2. num_trials_per_task default: Spec says 50 episodes. run_libero_eval.py hardcodes range(50) in run_task() (line 448), not using
 cfg.num_trials_per_task. New script should use cfg.num_trials_per_task correctly.
 3. Checkpoint: The default checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 — confirm it covers all 4 spatial
 tasks (name suggests spatial+object+goal; long suite may need a different checkpoint).
 4. safety_level parameter type: Currently str = "I" in config. Confirm benchmark dict accepts "I" vs 1 vs "level_I".
 5. initial_states_path: The spec shows 50 episodes per task. With DEFAULT, task_suite.get_task_init_states(task_id) must return >= 50
 states. Confirm this.
 