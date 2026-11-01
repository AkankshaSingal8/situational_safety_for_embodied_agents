# Inspection record

Date: 2026-10-11. Root repository HEAD: `75ca90d4`.

The working tree had existing local changes in SafeLIBERO, openvla-oft, and vlsa-aegis. Those were not changed. Earlier proposal documents were preserved. No commits, pushes, training jobs, or rollout jobs were performed.

Read-only observations:

- `sinfo -h -o '%P|%G|%l'`: GPU-shared advertises h100-80, l40s-48, v100-32, v100-16 resources. This lists configured resources, not immediately available capacity.
- `sacctmgr -n -P show assoc where user=asingal format=Account,Partition,QOS`: account `cis240028p`, QOS `ft,gpu,gpuinteract,low,unlimited`.
- `sbatch --test-only -A cis240028p -p GPU-shared --gres=gpu:h100-80:1 -t 48:00:00 --wrap=true`: accepted with an estimated start/node. Test-only does not submit a job. It does not verify remaining SU balance or guarantee a start time.
- Account-specific SU balance was not checked. The budget in the proposal is an explicit planning estimate, conditional on pilot timing.

Code inspected: README.md, CLAUDE.md, fol_safety_filter/{filter,kb,rule_composer,cbf_mapper,visual_grounder}.py, vlm_pipeline/qwen_vlm_server.py, existing Slurm scripts, docs/{psc_workflow,code_review_findings,safety_memory_research,safety_memory_high_risk_proposal}.md, earlier Overleaf source, and experiments/safety_memory_poc/README.md.

The safety-metric issues are documented repository findings, not a new audit of the current benchmark submodules. The design explicitly requires independent validation before using those metrics as labels.

Literature review: targeted primary-source review through 2026-10-11, with abstract/project-page screening and closer method inspection for RoboSafe and FailBank. It is not an exhaustive systematic review or an independently reproduced evaluation. Bibliography metadata was read from primary arXiv records where available; exceptions are identified in sources.json and the bibliography.
