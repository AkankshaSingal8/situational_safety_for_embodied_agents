"""Verified residual data (v2): pi0.5 LIBERO + task-aware safety filter, kept only when it ends in safe success.

    python collect_v2.py --suite safelibero_spatial --ckpt runs_scale/spatial/base/ema_last.pt --base pi05 \
        --scenes data_v2/spatial/scenes_train.npz --cell_cfgs runs_v2/spatial/cell_cfgs.json --out data_v2/spatial/r0

Per scene, one driving sentence names a present obstacle (p = 0.8) or nothing.
  named   : run pi0.5 + safety_filter with the cell's best config; if the episode does not end with task success
            and the named obstacle unmoved (> 1 mm displacement = collision, as in the benchmark), retry with the
            cell's next config, up to --tries. Kept only on safe success; otherwise only the log line is written.
  nothing : run pi0.5 alone; always kept (its rows teach "no sentence -> no correction").
With --residual (DAgger), the executed chunk is pi0.5 + the residual for the driving sentence, the filter's
plan/CBF still produce the labels, and the episode is kept only on safe success.

One npz per kept episode, one row per executed chunk:
  agent, wrist  (C,224,224,3) uint8   env frames (raw orientation; pi0.5 itself sees them flipped 180 deg)
  state         (C,8)                 eef pos, axis-angle, gripper qpos (pi0.5's state)
  base, plan    (C,H,7)               pi0.5 chunk and the filter's planned chunk (env units)
  exec          (C,k,7)               actions actually executed (after the per-step CBF)
  drive_tgt     (C,H,3)               label for the driving sentence: executed - base on the first k steps,
                                      planned - base on the rest; zeros when nothing is named
  other_tgt     (C,M,H,3)             filter plan - base for each other present obstacle (not rollout-verified)
  meta          json string           scene, layout, level, task, lang, drive, others, config, try, outcome
Labels are in env action units; the trainer normalizes.
"""
import argparse
import json
import os
import time

import numpy as np


def read_log_lines(path):
    """Parsed lines of one jsonl log; skips lines garbled by concurrent appends (older runs shared log names)."""
    out = []
    for x in open(path, errors="replace"):
        try:
            out.append(json.loads(x.replace("\x00", "").strip()))
        except ValueError:
            pass
    return out


def state8(obs):
    from pi05_runner import quat2axisangle
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"])).astype(np.float32)


def run_one(env, init, lang, r, cfg, drive, others, fcfg, res=None, temb=None):
    """One episode. Returns (outcome dict, per-chunk record lists)."""
    import cv2
    import torch  # noqa: F401
    import sl_env
    from shield import make_filter
    env.reset()
    obs = env.set_init_state(init)
    obs = sl_env.settle(env)
    present = sl_env.active_obstacles(obs)
    init_pos = {n: obs[n + "_pos"].copy() for n in present}
    k, tid = cfg["exec_steps"], r.task_id(lang)
    filt = make_filter(fcfg) if drive else None
    if filt:
        filt.reset(env, obs, drive)
    oth = {}
    for n in others:
        f = make_filter(fcfg)
        f.reset(env, obs, [n])
        oth[n] = f
    half = (r.amax[:3] - r.amin[:3]) / 2
    rec = {key: [] for key in ("agent", "wrist", "state", "base", "plan", "exec", "drive_tgt", "other_tgt")}
    hits, t, success, queue = {}, 0, False, []
    exec_buf = None
    while t < sl_env.MAX_STEPS:
        if not queue:
            if exec_buf is not None:
                rec["exec"].append(np.array(exec_buf, np.float32))
            x, f = r.chunk(obs, tid)
            A = r.unnorm(x.cpu().numpy())
            plan = filt.plan_chunk(env, obs, A) if filt else A.copy()
            run_A = plan
            if res is not None:  # DAgger: the residual drives, the filter only labels
                d = res.sample(f, x.unsqueeze(0), temb, steps=cfg["flow_steps"])[0].cpu().numpy() / cfg["res_scale"]
                run_A = A.copy()
                run_A[:, :3] = np.clip(A[:, :3] + d * half, -1, 1)
            ot = []
            for n, of in oth.items():
                ot.append(of.plan_chunk(env, obs, A)[:, :3] - A[:, :3])
            rec["agent"].append(cv2.resize(obs["agentview_image"], (224, 224), interpolation=cv2.INTER_AREA))
            rec["wrist"].append(cv2.resize(obs["robot0_eye_in_hand_image"], (224, 224), interpolation=cv2.INTER_AREA))
            rec["state"].append(state8(obs))
            rec["base"].append(A.astype(np.float32)); rec["plan"].append(plan.astype(np.float32))
            rec["other_tgt"].append(np.array(ot, np.float32).reshape(len(oth), len(A), 3))
            queue, exec_buf = list(run_A[:k]), []
        a = queue.pop(0)
        if filt:
            a = filt.step(env, obs, a)
        for of in oth.values():  # keep the label-only filters' grasp state in sync with what is executed
            of.grip_cmd = float(a[6])
        exec_buf.append(np.asarray(a, np.float32))
        obs, _, done, _ = env.step(np.asarray(a).tolist())
        for n in sl_env.collided(obs, init_pos, [n for n in present if n not in hits]):
            hits[n] = t
        if done:
            success = True
            break
        t += 1
    if exec_buf:
        rec["exec"].append(np.array(exec_buf, np.float32))
    # labels for the driving sentence: what was executed on the executed steps, the plan on the rest
    for c in range(len(rec["base"])):
        A, P, E = rec["base"][c], rec["plan"][c], rec["exec"][c]
        tgt = np.zeros((len(A), 3), np.float32)
        if drive:
            tgt[:] = P[:, :3] - A[:, :3]
            tgt[:len(E)] = E[:, :3] - A[:len(E), :3]
        rec["drive_tgt"].append(tgt)
    named_hit = any(sl_env.scene_name(obs, n) in hits for n in drive)
    out = dict(success=bool(success), hits=hits, named_hit=bool(named_hit), steps=t,
               safe_success=bool(success and not named_hit), filter=filt.stats if filt else None)
    return out, rec


def save(path, rec, meta):
    exec_ = np.zeros((len(rec["exec"]), max(len(e) for e in rec["exec"]), 7), np.float32)
    for i, e in enumerate(rec["exec"]):
        exec_[i, :len(e)] = e
    np.savez_compressed(path, agent=np.stack(rec["agent"]), wrist=np.stack(rec["wrist"]),
                        state=np.stack(rec["state"]), base=np.stack(rec["base"]), plan=np.stack(rec["plan"]),
                        exec=exec_, exec_len=np.array([len(e) for e in rec["exec"]], np.int16),
                        drive_tgt=np.stack(rec["drive_tgt"]), other_tgt=np.stack(rec["other_tgt"]),
                        meta=json.dumps(meta))


def worker(args):
    wid, ids, cfg = args
    import torch
    torch.set_num_threads(1)
    import sl_env
    sl_env.set_suite(cfg["suite"])
    sl_env.ENV_SEED, sl_env.SETTLE = cfg["env_seed"], cfg["settle"]
    from controllers import make_runner
    from eval_sl import load_scenes
    states, meta = load_scenes(cfg["scenes"])
    grid = json.load(open(cfg["grid"]))
    cells = json.load(open(cfg["cell_cfgs"])) if cfg["cell_cfgs"] else {}
    r = make_runner(cfg)
    res = bank = None
    if cfg.get("residual"):
        import texts as TX
        from policy import ResidualFlow
        ck = torch.load(cfg["residual"], map_location="cpu")
        res = ResidualFlow(feat_dim=ck["feat_dim"], text_dim=768 if ck["variant"] == "text" else 0,
                           horizon=ck.get("horizon", 16))
        res.load_state_dict(ck["model"]); res.eval().to(cfg["device"]); cfg["res_scale"] = ck["scale"]
        bank = TX.TextBank()
    rng = np.random.default_rng(cfg["seed"] * 1000 + wid)
    log = open(os.path.join(cfg["out"], f"log_{wid}_{os.environ.get('SLURM_JOB_ID', 'local')}.jsonl"), "a")
    done = set()
    for f in os.listdir(cfg["out"]):
        if f.startswith("log_"):
            done |= {x["scene"] for x in read_log_lines(os.path.join(cfg["out"], f)) if x.get("final")}
    envs = {}
    for i in ids:
        if i in done:
            continue
        m = meta[i]
        key = (m["level"], m["task"])
        if key not in envs:
            for v in envs.values():
                v[0].close()
            envs = {key: sl_env.make_env(*key, res=cfg["res"])}
        env, lang, _ = envs[key]
        srng = np.random.default_rng(cfg["seed"] * 100003 + i)  # driving choice depends only on the scene
        env.reset(); obs = env.set_init_state(states[i]); obs = sl_env.settle(env)
        present = [n for n in m["obstacles"] if n in sl_env.active_obstacles(obs)]
        drive = [present[srng.integers(len(present))]] if (present and srng.random() < cfg.get("p_drive", 0.8)) else []
        others = [n for n in present if n not in drive]
        if cfg.get("retry_drive") and cfg["retry_drive"].get(str(i)) != drive:
            log.write(json.dumps(dict(scene=i, drive=drive, final=True, keep=False, retry_mismatch=True,
                                      expected=cfg["retry_drive"].get(str(i)))) + "\n"); log.flush()
            continue
        temb = None
        if res is not None:
            import texts as TX
            sent = ""
            if drive:
                words = TX.train_texts(sl_env.canon(drive[0]))
                sent = words[srng.integers(len(words))]
            temb = torch.from_numpy(bank(sent)).unsqueeze(0).to(cfg["device"])
        names = cells.get(f"{m['level']}/{m['task']}", ["default"]) if drive else ["base"]
        names = names[: cfg["tries"]]
        if cfg.get("retry_cfgs") and drive:
            names = cfg["retry_cfgs"].split(",")
        for k_try, name in enumerate(names):
            fcfg = grid.get(name, {})
            try:
                out, rec = run_one(env, states[i], lang, r, cfg, drive, others, fcfg, res, temb)
            except Exception as ex:
                out, rec = dict(error=repr(ex), safe_success=False, success=False), None
            keep = rec is not None and (out["safe_success"] if (drive or res is not None) else True)
            last = keep or k_try == len(names) - 1
            line = dict(scene=i, layout=m["layout"], level=m["level"], task=m["task"], drive=drive, others=others,
                        cfg=name, attempt=k_try, keep=bool(keep), final=bool(last), **out)
            log.write(json.dumps(line) + "\n"); log.flush()
            if keep:
                save(os.path.join(cfg["out"], f"ep_{i:05d}.npz"), rec,
                     dict(line, lang=lang, exec_steps=cfg["exec_steps"], suite=cfg["suite"]))
            if last:
                break
    return wid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the suite's flow base (pi0.5 runner uses its action range)")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--base", default="pi05", choices=["flow", "pi05"])
    ap.add_argument("--pi_port", type=int, default=8000)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--env_seed", type=int, default=0)
    ap.add_argument("--settle", type=int, default=20)
    ap.add_argument("--exec_steps", type=int, default=5)
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument("--grid", default="filter_grid.json")
    ap.add_argument("--cell_cfgs", default="", help="json {'I/0': [cfg names, best first], ...}")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene_ids", default="")
    ap.add_argument("--p_drive", type=float, default=0.8, help="chance a scene gets a driving sentence (r1: 1.0)")
    ap.add_argument("--residual", default="")
    ap.add_argument("--retry_from", default="", help="earlier output dir: rerun only its named scenes that ended "
                    "without a safe success (same seed -> same scene and driving obstacle), writing to --out")
    ap.add_argument("--retry_cfgs", default="shield_h7,shield_h5,shield_wide_h7,shield_h7",
                    help="filter configs tried in order for --retry_from scenes")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from eval_sl import load_scenes
    n = len(load_scenes(a.scenes)[1])
    lo, hi = map(int, a.scene_ids.split("-")) if a.scene_ids else (0, n - 1)
    ids = list(range(lo, min(hi, n - 1) + 1))
    if a.retry_from:
        fin = {}
        for f in os.listdir(a.retry_from):
            if f.startswith("log_"):
                for x in open(os.path.join(a.retry_from, f), errors="replace"):
                    try:
                        x = json.loads(x.replace("\x00", "").strip())
                    except ValueError:
                        continue
                    if x.get("final"):
                        fin[x["scene"]] = x
        ids = [i for i in ids if i in fin and fin[i]["drive"] and not fin[i]["keep"]]
        a.retry_drive = {str(i): fin[i]["drive"] for i in ids}   # checked per scene (same seed -> same choice)
        print(f"retry: {len(ids)} named scenes without a safe success in {a.retry_from}", flush=True)
    else:
        a.retry_cfgs, a.retry_drive = "", {}
    per = int(np.ceil(len(ids) / a.workers))
    blocks = [ids[j * per:(j + 1) * per] for j in range(a.workers)]
    cfg = vars(a)
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for wid in pool.imap_unordered(worker, [(j, b, cfg) for j, b in enumerate(blocks) if b]):
            print(f"worker {wid} done {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = [x for f in os.listdir(a.out) if f.startswith("log_") for x in read_log_lines(os.path.join(a.out, f))]
    fin = [x for x in rows if x["final"]]
    dr = [x for x in fin if x["drive"]]
    print("retry drive mismatches:", sum(bool(x.get("retry_mismatch")) for x in fin), flush=True)
    print(f"scenes {len(fin)}; named {len(dr)}: kept {sum(x['keep'] for x in dr)} "
          f"({np.mean([x['keep'] for x in dr]) if dr else float('nan'):.3f}), attempts {len([x for x in rows if x['drive']])}; "
          f"base-only {len(fin) - len(dr)}", flush=True)


if __name__ == "__main__":
    main()
