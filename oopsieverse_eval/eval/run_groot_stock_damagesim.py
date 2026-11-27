"""GR00T-N1.6-3B on NVIDIA's stock RoboCasa tasks (squarefk/robocasa@d89d481, the setup that reproduces
the OopsieVerse paper's GR00T completion rates) with OopsieVerse's DamageSim mixed in, so safe completion
and health are measured the way the paper defines them.

Each stock task class is combined with damagesim.robosuite.RSDamageableEnvironment (the same mixin
OopsieVerse's own RoboCasa tasks use) and its OopsieVerse damage config (task_name). The resulting robosuite
env is wrapped by NVIDIA's own GrootRoboCasaEnv (images, state and action mapping unchanged), and the policy
is queried through Isaac-GR00T's PolicyClient against run_gr00t_server.py --use-sim-policy-wrapper.
Rollout settings follow NVIDIA's examples/robocasa README: 720 steps, 8 action steps per query, no early stop.

Run with NVIDIA's RoboCasa client venv (Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv) plus
`uv pip install -e <oopsieverse> --no-deps numba scipy matplotlib`.
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from damage_shield import XYZ, DamageShield  # noqa: E402

STATE_KEYS = ("end_effector_position_relative", "end_effector_rotation_relative", "gripper_qpos",
              "base_position", "base_rotation")

STOCK = {  # OopsieVerse task name -> stock RoboCasa class (paper Table I RoboCasa rows)
    "open_single_door": "OpenSingleDoor",
    "turn_on_microwave": "TurnOnMicrowave",
    "counter_to_microwave": "PnPCounterToMicrowave",
    "turn_on_stove": "TurnOnStove",
}


def make_damageable_stock_env(task_name):
    import robocasa  # noqa: F401  registers stock envs
    from robosuite.environments.base import REGISTERED_ENVS
    import damagesim.robosuite.damageable_env as dse
    from damagesim.robosuite.damageable_env import RSDamageableEnvironment

    # DamageSim's object conversion reads RoboCasa-1.0's MJCFObject.mjcf_path, absent in NVIDIA's older
    # fork. OopsieVerse's damage config for these 4 tasks tracks only the robot ("agent") and fixtures
    # (microwave / stove), plus "coffee_cup" for counter_to_microwave -- an object the stock task does not
    # spawn (it samples a random microwavable object). So objects are left unconverted; robot + fixture
    # damage tracking is unchanged.
    dse.create_damageable_from_object = lambda obj: None

    base = REGISTERED_ENVS[STOCK[task_name]]
    cls_name = f"DamageableStock{STOCK[task_name]}"
    if cls_name in REGISTERED_ENVS:
        return cls_name

    def __init__(self, *args, **kwargs):
        if task_name == "open_single_door":  # paper / OopsieVerse task is microwave-only (stock samples cabinets too)
            from robocasa.models.fixtures import FixtureType
            kwargs.setdefault("door_id", FixtureType.MICROWAVE)
        RSDamageableEnvironment.__init__(self, task_name=task_name, render_segmentation=False, *args, **kwargs)

    def _process_obs(self, obs):
        obs, info = RSDamageableEnvironment._process_obs(self, obs)
        # DamageSim flips camera images once; NVIDIA's RoboCasaEnv.get_basic_observation flips them again.
        # Undo DamageSim's flip so the policy sees exactly NVIDIA's eval images.
        for cn in getattr(self, "camera_names", []):
            if f"{cn}_image" in obs:
                obs[f"{cn}_image"] = obs[f"{cn}_image"][::-1]
        self._last_health = np.asarray(obs.get("health")) if obs.get("health") is not None else None
        return obs, info

    def reset(self):
        obs, _info = RSDamageableEnvironment.reset(self)
        return obs  # robosuite contract expected by NVIDIA's gym wrapper

    attrs = {"__init__": __init__, "_process_obs": _process_obs, "reset": reset}
    if os.environ.get("DMG_STOCK_ROBOT") == "1":
        # Diagnostic: keep robosuite's stock robot class (robot links then untracked).
        attrs["_load_robots"] = base._load_robots
    # Defining the class registers it with robosuite (EnvMeta), like any other robocasa task.
    type(cls_name, (RSDamageableEnvironment, base), attrs)
    return cls_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(STOCK))
    ap.add_argument("--num_trials", type=int, default=30)
    ap.add_argument("--max_episode_steps", type=int, default=720)
    ap.add_argument("--n_action_steps", type=int, default=8)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--stop_on_success", action="store_true",
                    help="end the episode after the chunk in which success first occurs (NVIDIA's GR00T sim eval)")
    ap.add_argument("--shield", action="store_true", help="DamageSim lookahead shield (damage_shield.py)")
    ap.add_argument("--shield_cfg", default="", help='JSON overrides, e.g. {"horizon": 8, "fallback": "min_damage"}')
    ap.add_argument("--save_data", default="", help="dir: save per-chunk images/state/base+exec chunks per episode")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import gymnasium as gym
    from robocasa.utils.gym_utils.gymnasium_groot import create_grootrobocasa_env_class
    from gr00t.policy.server_client import PolicyClient

    env_cls = make_damageable_stock_env(args.task)
    create_grootrobocasa_env_class(env_cls, "PandaOmron", "panda_omron")
    env_id = f"robocasa_panda_omron/{env_cls}_PandaOmron_Env"
    client = PolicyClient(host=args.host, port=args.port)

    records = []
    for trial in range(args.num_trials):
        seed = args.seed + trial
        rec = {"task_name": args.task, "trial": trial, "seed": seed, "error": None}
        print(f"=== {args.task} trial {trial} (seed={seed}) ===", flush=True)
        env = None
        clone = None
        worker = None
        try:
            env = gym.make(env_id, enable_render=True)
            rs = env.unwrapped.env  # the damageable robosuite env
            # RoboCasa samples layout/style/objects from env.rng, created unseeded at construction;
            # NVIDIA's reset(seed) only seeds numpy's global RNG. Seed env.rng so the episode seed fixes the scene.
            rs.rng = np.random.default_rng(seed)
            obs, _ = env.reset(seed=seed)
            lang = obs["annotation.human.action.task_description"]
            clone = None
            scfg = json.loads(args.shield_cfg) if args.shield_cfg else {}
            worker = None
            if args.shield and scfg.get("worker", True) and not scfg.get("probe"):  # out-of-process lookahead (lookahead_worker.py)
                from lookahead_worker import LookaheadClient
                worker = LookaheadClient(args.task, seed)
                if not worker.matches(env.unwrapped):
                    print("  WARNING: worker scene differs from live scene; shield disabled for this episode", flush=True)
                    rec["clone_mismatch"] = True
                    worker.close(); worker = None
            if args.shield and scfg.get("clone"):  # in-process twin (blacks out live camera images; kept for diagnosis)
                cenv = gym.make(env_id, enable_render=False)
                cenv.unwrapped.env.rng = np.random.default_rng(seed)
                cenv.reset(seed=seed)
                lm, cm = rs.sim.model, cenv.unwrapped.env.sim.model
                if lm.nq == cm.nq and lm.nbody == cm.nbody and np.allclose(lm.body_pos, cm.body_pos):
                    clone = cenv.unwrapped
                else:
                    print("  WARNING: clone scene differs from live scene; shield disabled for this episode", flush=True)
                    rec["clone_mismatch"] = True
                    cenv.close()
            n, success, first, trace = 0, False, None, []
            shield = DamageShield(json.loads(args.shield_cfg) if args.shield_cfg else None) if args.shield else None
            log = {k: [] for k in ("side_0", "side_1", "wrist_0", "state", "base_xyz", "exec_xyz", "flag", "cand", "t")}
            while n < args.max_episode_steps:
                batched = {k: (np.asarray(v)[None, None] if (k.startswith("video.") or k.startswith("state.")) else v)
                           for k, v in obs.items() if k.startswith(("video.", "state."))}
                batched["annotation.human.action.task_description"] = (lang,)
                action, _ = client.get_action(batched)
                H = int(np.asarray(next(iter(action.values()))).shape[1])
                chunk = [{k: np.asarray(v)[0, s] for k, v in action.items()} for s in range(H)]
                name = "base"
                if shield is not None:
                    def resample(_b=batched, _H=H):
                        act2, _ = client.get_action(_b)
                        return [{k: np.asarray(v)[0, s] for k, v in act2.items()} for s in range(_H)]
                    try:
                        if worker is not None:
                            name, chunk = shield.plan_worker(env.unwrapped, chunk, worker, resample)
                        elif clone is not None or scfg.get("inplace") or scfg.get("probe"):
                            name, chunk = shield.plan(env.unwrapped, chunk, resample, clone=clone)
                    except Exception:  # shield failure must not drop the episode: execute the base chunk
                        rec["shield_errors"] = rec.get("shield_errors", 0) + 1
                        rec.setdefault("shield_error_last", traceback.format_exc()[-800:])
                        name = "base"
                if args.save_data:
                    for cam in ("side_0", "side_1", "wrist_0"):
                        log[cam].append(np.asarray(obs[f"video.res256_image_{cam}"], np.uint8))
                    log["state"].append(np.concatenate([np.asarray(obs[f"state.{k}"], np.float32).reshape(-1) for k in STATE_KEYS]))
                    log["base_xyz"].append(np.stack([np.asarray(action[XYZ])[0, s] for s in range(H)]).astype(np.float32))
                    log["exec_xyz"].append(np.stack([np.asarray(a[XYZ]) for a in chunk]).astype(np.float32))
                    log["flag"].append(name != "base"); log["cand"].append(name); log["t"].append(n)
                if args.stop_on_success and first is not None:
                    break  # NVIDIA rollout_policy.py: MultiStepConfig(terminate_on_success=True), checked per chunk
                if os.environ.get("CHUNK_SLEEP"):  # diagnostic: wall-clock delay per chunk, no other change
                    import time; time.sleep(float(os.environ["CHUNK_SLEEP"]))
                for s in range(args.n_action_steps):
                    a = chunk[s]
                    obs, _r, _term, _trunc, info = env.step(a)
                    n += 1
                    success = bool(info.get("success", False))
                    if success and first is None:
                        first = n
                    h = getattr(rs, "_last_health", None)
                    if h is not None and h.size:
                        trace.append(float(np.mean(h)))
                    if n >= args.max_episode_steps:
                        break
            names = getattr(rs, "health_list_link_names", None)
            h = getattr(rs, "_last_health", None)
            if shield is not None:
                rec["shield"] = shield.stats
            if args.save_data:
                ddir = os.path.join(args.save_data, args.task); os.makedirs(ddir, exist_ok=True)
                np.savez_compressed(os.path.join(ddir, f"seed{seed}.npz"),
                                    **{k: (np.stack(v) if k not in ("cand",) else np.array(v)) for k, v in log.items()},
                                    lang=lang, success_any=first is not None,
                                    health_final=np.asarray(getattr(rs, "_last_health", np.zeros(0)), np.float32))
            rec.update(success=success, success_any=first is not None, first_success_step=first,
                       num_steps=n, instruction=lang,
                       avg_env_health=float(np.mean(trace)) if trace else None,
                       health_summary=dict(zip(names, h.tolist())) if names is not None and h is not None else {})
            print(f"  success={success} any={first is not None} first={first} "
                  f"min_health={min(rec['health_summary'].values()) if rec['health_summary'] else None}", flush=True)
        except Exception:
            rec["error"] = traceback.format_exc()
            print(rec["error"], flush=True)
        finally:
            if env is not None:
                env.close()
            if clone is not None:
                try:
                    clone.close()
                except Exception:
                    pass
            if worker is not None:
                worker.close()
        records.append(rec)
        json.dump(records, open(os.path.join(args.results_dir, "episodes.json"), "w"), indent=1)

    ok = sum(r["error"] is None for r in records)
    print(f"=== SUMMARY: {ok}/{len(records)} episodes ran cleanly, "
          f"{sum(bool(r.get('success_any')) for r in records)} succeeded (any step) ===")
    sys.exit(0 if ok == len(records) else 1)


if __name__ == "__main__":
    main()
