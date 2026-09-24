# Dreamer-v2-revisited
# UNDER PROGRESS

A re-implementation of Dreamer v2 written in 2026 for learning purposes, built on top of [Dreamer-v1-revisited](https://github.com/Farag-Y/Dreamer-v1-revisited) (which itself builds on the world model from [PlaNet-revisited](https://github.com/Farag-Y/PlaNet-revisited)), using a modern Python project structure (uv, Hydra config, modular model layout).

---

## What is Dreamer v2?

Dreamer v2 ([Mastering Atari with Discrete World Models](https://arxiv.org/pdf/2010.02193), Hafner et al. 2020) is a model-based reinforcement learning agent that learns entirely from imagined trajectories inside a learned world model, and was the first agent to reach human-level performance on the Atari benchmark purely from imagined rollouts, without a lookahead search.

It keeps Dreamer v1's recipe — a **Recurrent State Space Model (RSSM)** that learns latent dynamics from pixels, and an actor-critic trained by backpropagating through imagined latent trajectories — but reworks the world model's representation and training to make imagination accurate enough for the harder, discrete-action Atari setting:

- **Categorical latent states:** the stochastic part of the RSSM is a vector of several categorical variables (rather than a single diagonal Gaussian), sampled with a straight-through gradient estimator so the model stays differentiable end-to-end.
- **KL balancing:** the world model's KL term is split into separate weights for training the prior towards the posterior and the posterior towards the prior, so the dynamics model learns to predict the future without the posterior being regularized as aggressively.
- **Image, reward, and discount predictors:** as in Dreamer v1, a CNN encoder/decoder reconstructs pixels, and separate dense heads predict reward and episode continuation (discount), all trained jointly with the dynamics.
- **Actor-critic in latent imagination:** the actor and value model are trained purely on trajectories imagined inside the RSSM, using a mix of reinforce and straight-through gradients suited to discrete Atari actions, with returns computed via TD(λ)-style targets bootstrapped by the value model.
- **Single GPU, single environment instance:** unlike prior Atari agents that relied on large-scale distributed data collection, Dreamer v2 reaches human-level performance training on a single GPU with a single environment.

---

## This Re-implementation

This repo starts from the [Dreamer-v1-revisited](https://github.com/Farag-Y/Dreamer-v1-revisited) codebase — same RSSM, encoder, observation model, reward model, and latent-imagination actor-critic — and is being extended towards Dreamer v2's world model. Current state:

- A **discount/continuation predictor** (`models/discount_model.py`) has been added to the world model and is used both to end imagined rollouts early and to bootstrap λ-returns for envs that can terminate early (see `TERMINATING_ENVS` in `env_registry.py`).
- The RSSM's stochastic state is still a diagonal Gaussian trained with free-nats, as in Dreamer v1 — swapping it for Dreamer v2's categorical latents with KL balancing and straight-through gradients is the main piece of work still ahead.
- The actor and critic are unchanged from Dreamer v1: a continuous, tanh-squashed Gaussian policy trained by backpropagating analytic gradients through imagined rollouts (rather than the reinforce-based estimator Dreamer v2 uses for discrete Atari actions), since this repo targets continuous-control benchmarks (MuJoCo, Box2D, dm_control) rather than Atari.

Other goals, unchanged from the v1 repo:

- Clean, readable code that maps closely to the paper.
- Modern Python tooling: [uv](https://docs.astral.sh/uv/) for environment management, [Hydra](https://hydra.cc) for configuration.
- Modular layout: each model component (RSSM, encoder, observation model, reward model, discount model, actor/critic) lives in its own file under `models/`.
- Uses `gymnasium` (the maintained fork of OpenAI Gym) instead of the original `gym`.
- Supports `dm_control` environments alongside gymnasium.

It does **not** aim to reproduce the exact benchmark numbers from the paper.

---

## Project Structure

```
Dreamer-v2-revisited/
├── conf/
│   └── config.yaml         # All hyperparameters via Hydra
├── models/
│   ├── rssm.py              # Recurrent State Space Model (world model)
│   ├── encoder.py
│   ├── observation_model.py
│   ├── reward_model.py
│   ├── discount_model.py    # Episode-continuation ("pcont") predictor
│   └── actor_critic.py      # Actor + Critic trained in latent imagination
├── agent/
│   ├── world_model.py       # World model training loss (KL, image, reward, discount)
│   ├── behavior.py           # Imagined rollouts + actor-critic training
│   └── dreamer.py            # Ties world model + behavior together, env interaction loop
├── env_wrapper.py            # Gymnasium + dm_control wrappers (image preprocessing)
├── experience_replay.py      # Replay buffer
├── checkpoint.py             # Checkpointing
├── cloud_storage.py          # Cloudflare R2 upload/listing helpers
├── main.py                   # Training entry point
├── play.py                   # Manual keyboard control of any supported env
└── utils.py
```

---

## Installation

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone <repo-url>
cd Dreamer-v2-revisited
uv sync
```

Run:

```bash
uv run python main.py
```

Override any config value inline:

```bash
uv run python main.py env=HalfCheetah-v5 seed=42
```

---

## Environments

Two environment families are supported. Set `env` in `conf/config.yaml` or via the command line.

**Gymnasium** — install with `uv sync` (included by default):

| Category | Examples |
|---|---|
| Classic Control | `Pendulum-v1`, `MountainCarContinuous-v0` |
| Box2D | `BipedalWalker-v3`, `BipedalWalkerHardcore-v3`, `CarRacing-v3` |
| MuJoCo | `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Humanoid-v5`, ... |

**dm_control** — also included by default:

| Environment | dm_control task |
|---|---|
| `cartpole-swingup` | Cartpole swingup from hanging position |
| `finger-spin` / `finger-turn-easy` / `finger-turn-hard` | Robotic finger spinning / turning a body |
| `cheetah-run` | Half-cheetah running |
| `reacher-easy` / `reacher-hard` | 2-link arm reaching |
| `cup-catch` | Ball-in-cup |
| `walker-stand` / `walker-walk` / `walker-run` | Bipedal walker |
| `hopper-stand` / `hopper-hop` | Hopping leg |
| `humanoid-stand` / `humanoid-walk` / `humanoid-run` | Humanoid locomotion |

```bash
uv run python main.py env=cartpole-swingup
uv run python main.py env=finger-turn-hard
```

Environments with a natural early-termination signal (e.g. `Hopper-v5`, `BipedalWalker-v3`, `MountainCarContinuous-v0` — see `TERMINATING_ENVS` in `env_registry.py`) automatically enable the discount predictor during training.

> **macOS note:** dm_control rendering uses mujoco's native CGL renderer and does not require a system OpenGL installation.

---

## Playing Environments Manually

`play.py` lets you control any supported environment yourself using the keyboard. Useful for getting a feel for an environment before training.

```bash
uv run python play.py env=Pendulum-v1
uv run python play.py env=cartpole-swingup
```

A pygame window opens showing the environment at full resolution. Key bindings are displayed as an overlay at the bottom of the window and printed to the terminal on startup. Up to 4 action dimensions are mapped:

| Keys | Action dim |
|---|---|
| `←` / `→` | action\[0\] |
| `↑` / `↓` | action\[1\] |
| `A` / `D` | action\[2\] |
| `W` / `S` | action\[3\] |

Hold a key to push the action to its maximum; release to return to zero. `R` resets the episode, `Q` quits.

---

## Training on Vast.ai

`scripts/train_vastai.py` (run via `make train-vast`) is an interactive helper that rents a GPU on [Vast.ai](https://vast.ai), uploads the project, runs training, streams logs, and **automatically destroys the instance** when training finishes.

### Prerequisites

1. Install the Vast.ai CLI (requires the `vastai` dependency group):

   ```bash
   uv sync --group vastai
   ```

2. Authenticate:

   ```bash
   vastai set api-key <YOUR_KEY>
   ```

   Get your key at <https://cloud.vast.ai/account/>.

3. Add your API key to `.env` in the project root (see [`.env.example`](.env.example)):

   ```
   VAST_API_KEY=<your_key>
   ```

   The key is baked into the remote runner so the instance can self-destruct via the API when training ends.

### Usage

```bash
make train-vast
```

The script walks you through four interactive prompts:

| Prompt | Options |
|---|---|
| GPU type | RTX 3060, RTX 4090, RTX 3090, A100 SXM4 80GB, H100 NVL, A6000 |
| CUDA version | 12.1 or 12.4 |
| Entrypoint | full training run or evaluation only |
| Extra Hydra overrides | e.g. `env=HalfCheetah-v5 seed=42` |
| Max price per hour | e.g. `0.50` |

It then lists the matching offers (sorted by price) and lets you pick one. A confirmation prompt is shown before any money is spent.

Pass `--auto` to skip the offer picker and confirmation and use the cheapest match, and `--keep-alive` to leave the instance running after training instead of destroying it:

```bash
make train-vast ARGS="--auto"
make train-vast ARGS="--auto --keep-alive"
```

### What it does (step by step)

1. **Preflight** — checks that `vastai` CLI is installed and authenticated, and that `VAST_API_KEY` is set in `.env`.
2. **Search offers** — queries Vast.ai for rentable instances matching your GPU, CUDA, and price constraints.
3. **Create instance** — provisions the selected offer with disk and SSH access using the chosen PyTorch Docker image.
4. **Wait for boot** — polls until the instance status is `running`.
5. **Upload code** — rsyncs the project to `/workspace/` on the instance, excluding `.git/`, `.venv/`, `outputs/`, `results/`, and caches.
6. **Install deps** — installs system OpenGL/EGL libraries needed by MuJoCo/dm-control, then runs `uv sync` on the instance.
7. **Upload runner** — generates a remote run script with your entrypoint and Hydra overrides baked in; it calls the Vast.ai API to destroy the instance when the run exits (unless `--keep-alive` is set).
8. **Launch detached** — starts training via `nohup` so it survives SSH disconnects.
9. **Stream logs** — tails `/workspace/training.log` live. Press `Ctrl-C` to detach — training continues server-side and the instance self-destructs when done.

To reconnect to a running instance after detaching:

```bash
ssh -p <PORT> -o StrictHostKeyChecking=no root@<HOST>
tail -f /workspace/training.log
```

The reconnect command is printed when you detach.

---

## Checkpoint Cleanup on R2

When `r2_enabled: true` in the config, training uploads checkpoints and experience replays to a Cloudflare R2 bucket as it runs, which can accumulate a lot of stale intermediate checkpoints over a long run. `scripts/r2_cleanup.py` (run via `make r2-cleanup`) scans the bucket and, for each run prefix, keeps only the highest-numbered checkpoint directory and experience replay file, deleting the rest (`config.yaml`, `metrics.png`, and `training.log` are never touched).

```bash
make r2-cleanup                                              # scan every run prefix in the bucket
make r2-cleanup ARGS="--prefix 2026-09-10_14-23-01"           # scan a single run
make r2-cleanup ARGS="--yes"                                  # skip the confirmation prompt
```

Requires `CF_R2_ACCOUNT_ID`, `CF_R2_ACCESS_KEY`, and `CF_R2_SECRET_KEY` in `.env` or the environment.

---

## References

- [Mastering Atari with Discrete World Models](https://arxiv.org/pdf/2010.02193) — Hafner et al., 2020 (Dreamer v2)
- [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603) — Hafner et al., 2020 (Dreamer v1)
- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) — Hafner et al., 2019 (PlaNet, the world model this repo builds on)
- [danijar/dreamerv2](https://github.com/danijar/dreamerv2) — original TensorFlow implementation
- [Farag-Y/Dreamer-v1-revisited](https://github.com/Farag-Y/Dreamer-v1-revisited) — this repo's base implementation (RSSM, encoder, observation/reward models, latent-imagination actor-critic)
- [Farag-Y/PlaNet-revisited](https://github.com/Farag-Y/PlaNet-revisited) — the original world model (RSSM, encoder, observation/reward models)
