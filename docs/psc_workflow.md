# Running long Claude Code sessions on PSC Bridges-2

Everything below was verified on `br012.bridges2.psc.edu` on 2026-09-14 for user
`asingal`. Numbers marked **verified** came from the cluster itself, not from the
PSC user guide — several of the guide's stated limits do not match this account.

## The shape of the workflow

```
your laptop ──ssh──▶ br012 (login node)
                       └─ tmux session "safety"   ← Claude Code lives here, 0 SU
                            └─ sbatch ──▶ GPU-shared compute node  ← all real compute
```

Claude Code is an idle-most-of-the-time Node process. It belongs on the login
node, where it costs nothing and survives indefinitely. Every python eval, VLA
rollout, or MuJoCo render goes to `sbatch` — those are what login-node reaping
targets, and what would burn SUs if you parked them in an `interact` shell.

## Why not `interact`

| | limit |
|---|---|
| `gpuinteract` QOS max walltime | **8:00:00** (verified: `sacctmgr show qos`) |
| `gpuinteract` max jobs / submits | **1 / 1** (verified) |
| interactive idle disconnect | 30 min (PSC user guide) |

An 8-hour ceiling with a 30-minute idle kill is the opposite of a long session.
`interact` is for debugging on a GPU, not for hosting your agent.

## Verified account facts

| Fact | Value | How it was checked |
|---|---|---|
| Usable slurm account | **`cis240028p` only** | `sacctmgr show assoc user=asingal` returns one row |
| `cis250185p` slurm | **unusable** | `sbatch --test-only -A cis250185p` → `Invalid account or account/partition combination` |
| Usable partitions | **GPU, GPU-shared** | `cis240028p` QOS list is `ft,gpu,gpuinteract,low,push,unlimited` — no RM QOS, so `-p RM-shared` → `Invalid qos specification` |
| Max batch walltime | **48:00:00** | `--test-only` accepts 48h, rejects 72h with `QOSMaxWallDurationPerJobLimit` |
| Partition `MaxTime` | `UNLIMITED` | misleading — the real cap is in the QOS, above |
| Compute-node internet | **full egress** | job 45975742 on `v016`: api.anthropic.com → 401 (reachable), github.com → 200, huggingface.co → 200 |
| Login-node internet | yes | api.anthropic.com → 401 |
| `~/.claude` | symlink → `/ocean/.../claude_home`, 1.2 G | already off the 25 G `/jet` home quota — leave it that way |
| `--autocompact 1000000` | accepted | `claude --autocompact 1000000 --version` → rc 0; `bogusvalue` → rc 1 with a range error, so the flag is validated, not ignored |
| Committed `#SBATCH` blocks | declare partition/gres/time/mem/logs, **no `-A`** | `grep '^#SBATCH' slurm/**` — they inherit the default account, which is the one that works, so they submit as-is |
| `/jet` home free | 9.5 G of 25 G | do not move session state back here |

Because compute nodes have real egress, `vlm_prompt_runner/` (Anthropic / OpenAI /
Gemini) works from inside batch jobs with no proxy configuration.

## Daily use

```bash
cc                # create-or-attach tmux "safety" + Claude Code in the repo
cc hardware       # same for contextual_safety_franka_arm
cc --where        # which login node am I on, what sessions exist here
psc-status        # node, queue, last 24h of jobs, disk  (--su adds the slow SU query)
psc-submit slurm/safelibero/eval_baseline_openvla_spatial_L1.slurm
psc-wait <jobid>  # block once until the job ends, then print its tail
```

`cc` is idempotent (`tmux new-session -A`): running it when a session already
exists attaches instead of creating a second one. It exports `MUJOCO_GL=egl`,
`TMPDIR`, `XDG_CACHE_HOME` and `HF_HOME` into the tmux environment so panes you
open later inherit them, and passes `--continue` when the directory has prior
session state so a reattach lands back in the same conversation.

If Claude exits — on purpose, or because it failed to start — the pane drops to
a login shell and prints the exit code rather than closing. A pane that vanishes
instantly would take the error message with it.

Detach with `Ctrl-b d`. Close the laptop, lose wifi, fly somewhere — the session
keeps running.

## The one thing that will break this: login-node pinning

**A tmux server is local to one login node.** `ssh bridges2.psc.edu` round-robins
between 128.182.81.29 and .30, so the next login can land you on a different node
where `tmux ls` shows nothing. The session is alive; you are just on the wrong
machine.

`br012.bridges2.psc.edu` resolves to 128.182.81.30 — this node's public address —
so it is directly ssh-able. Put this on **your laptop** in `~/.ssh/config`:

```
Host psc
    HostName br012.bridges2.psc.edu
    User asingal
    ServerAliveInterval 30
    ServerAliveCountMax 6
    TCPKeepAlive yes

Host psc-any
    HostName bridges2.psc.edu
    User asingal
```

Then `ssh psc` always returns to the node holding your tmux, and `cc` reattaches.
If PSC reboots br012 for maintenance the session is gone — `cc` then starts fresh
and `--continue` restores the conversation, which is why durable notes belong in
the repo rather than in scrollback.

## Two axes of "longer"

**Wall-clock** is what tmux + pinned ssh solve, above.

**Context window** is separate, and PSC cannot help with it. The levers are:
`--autocompact 1000000` (already baked into `cc`) so compaction fires as late as
the 1M window allows; delegating file-dump-heavy searches to subagents so their
output never enters the main transcript; and writing conclusions into
`docs/` and the memory index as you go, so a compaction or a fresh session loses
nothing that mattered.

## Native background sessions (alternative to tmux)

Claude Code has its own daemon mode, useful for running several agents at once:

```bash
claude --bg 'run the L1 spatial eval and summarize'   # prints a short id
claude agents            # list
claude logs <id>         # recent output
claude attach <id>       # take it over in this terminal
claude stop <id>         # stop; conversation kept, attach/resume still work
claude respawn --all     # restart all of them on a new Claude Code version
```

These daemons are still processes on **this login node**, so they need the same
ssh pinning. tmux remains the better primary: it keeps the TUI, survives ssh
death identically, and is one mental model rather than two.

## Long-running compute

`psc-submit` has two modes, and the distinction matters for reproducibility:

- **Script mode** (`psc-submit foo.slurm`) stamps **only `-A`**. Command-line
  sbatch options override in-script `#SBATCH`, so a helper that always passed
  `--gres=gpu:v100-32:1` would silently downgrade the committed h100 eval scripts
  and produce numbers that do not match the script in git. Flags you type *are*
  forwarded — that is an explicit override, not a default.
- **Wrap mode** (`psc-submit --wrap '<cmd>'`) applies the full defaults, since
  there is no script whose declarations need respecting.

```bash
# 48h is the hard ceiling; --time-min lets the backfill scheduler start you early
psc-submit -t 48:00:00 --time-min 04:00:00 slurm/libsafety/eval_flowcbf_libsafety_full_v2fix.slurm
```

`psc-wait` reads `StdOut=`/`StdErr=` out of `scontrol` before the job leaves the
queue, so it finds the log even when a script writes somewhere other than
`slurm_logs/` (several `libsafety/` scripts write into a `.worktrees/` path).

For anything longer than 48h, checkpoint and chain — `sbatch --dependency=afterok:<jobid>`.
Keep `env_seed` fixed across every job in a comparison table (see the repo
CLAUDE.md); resuming from a checkpoint must not reseed the simulator or the
object positions move.

Optional: `claude setup-token` issues a long-lived auth token. Whether OAuth
expiry actually interrupts an overnight session here has not been observed — it
is cheap insurance, not a diagnosed problem.

## Nothing multi-minute in the tmux pane

The repo's CLAUDE.md gives `python -m pytest fol_safety_filter/tests
vlm_prompt_runner/tests` as the test command. Run it through the helper, not in
the pane:

```bash
psc-submit --wrap 'cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents && python -m pytest fol_safety_filter/tests vlm_prompt_runner/tests'
```

Login nodes reap CPU-heavy processes. An idle Claude Code is fine; a test suite,
an eval, or a MuJoCo render in that same pane is what gets killed — and it takes
the whole pane with it.

## Files this workflow installs

| Path | Purpose |
|---|---|
| `~/bin/cc` | create-or-attach tmux + Claude Code |
| `~/bin/psc-submit` | sbatch with the verified account/partition/log defaults |
| `~/bin/psc-wait` | block once until a job finishes, print its tail |
| `~/bin/psc-status` | node, queue, recent jobs, disk, (opt-in) SU balance |
| `~/.tmux.conf` | 200k scrollback, `destroy-unattached off`, hostname in status bar |

`~/bin` is already on `PATH` via `.bashrc`.

## What was not testable from inside the cluster

`br012.bridges2.psc.edu` → 128.182.81.30 and port 22 is open on that address from
within PSC, so the `Host psc` block above should work — but it has not been tried
from an outside network. If `ssh psc` ever hangs, fall back to `ssh psc-any`, run
`cc --where` to see which node you landed on, and hop with `ssh br0NN`.
