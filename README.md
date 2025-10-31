# AIRL Pipeline Implementation

Clean, modular implementation of Adversarial Inverse Reinforcement Learning (AIRL) using the [HumanCompatibleAI/imitation](https://github.com/HumanCompatibleAI/imitation) library.

**Under Development**
---
For environments without pre-trained HuggingFace experts, custom expert training will be added in future versions.

**Currently tested environments**:
---
- `CartPole-v0` ✅
- `HalfCheetah-v1` ✅ (requires `gymnasium[mujoco]`)

---
## Features

- **Pre-trained Expert Loading**: Automatically loads pre-trained experts from HuggingFace
- **Vectorized Environments**: Parallel environment execution for faster training
- **Modular Pipeline**: Clear separation of setup → expert loading → demonstration collection → AIRL training
- **CLI Interface**: Easy configuration without code changes
- **Vector Observations Only**: Supports standard Gym environments (CartPole, HalfCheetah, etc.)

# Installation & Setup

### Windows (Command Prompt)

**First-time setup:**

```batch
# 1. Install Python 3.12 from python.org (if not installed)
python --version
# Should show: Python 3.12.x

# 2. Navigate to your AIRL folder
cd C:\Users\YourUsername\AIRL

# 3. Create a virtual environment
python -m venv venv

# 4. Activate environment
venv\Scripts\activate

# 5. Set version variable
set SETUPTOOLS_SCM_PRETEND_VERSION_FOR_IMITATION=0.0.0

# 6. Install the package
pip install -e .

# 7. For MuJoCo environments (HalfCheetah, etc.)
pip install gymnasium[mujoco]

# 8. Run your first test
python airl_pipeline.py --env CartPole-v0 
```

**Subsequent runs:**

```batch
# 1. Navigate to your AIRL folder
cd C:\Users\YourUsername\AIRL

# 2. Activate environment (previously installed)
venv\Scripts\activate

# 3. Run your first test
python airl_pipeline.py --env CartPole-v0 
```
### AWS/Cloud Environments [!! Read the Note !!]

**Note:** Untested - based on typical AWS EC2 setup patterns.

**First-time setup (Amazon Linux 2 / Ubuntu):**

```bash
# 1. Connect to your EC2 instance
ssh -i your-key.pem ec2-user@your-instance-ip

# 2. Install Python 3.12 (if not available, use Python 3.11)
sudo yum update -y                    # Amazon Linux
# OR
sudo apt update && sudo apt upgrade   # Ubuntu

# For Amazon Linux 2
sudo amazon-linux-extras install python3.11 -y
# OR for Ubuntu
sudo apt install python3.12 python3.12-venv -y

# 3. Upload your AIRL package to EC2 (from local machine)
# scp -i your-key.pem -r AIRL/ ec2-user@your-instance-ip:~/

# 4. Navigate to AIRL folder
cd ~/AIRL

# 5. Create virtual environment
python3.12 -m venv venv
# OR
python3.11 -m venv venv

# 6. Activate environment
source venv/bin/activate

# 7. Set environment variable
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_IMITATION=0.0.0

# 8. Install package
pip install -e .

# 9. For MuJoCo environments
pip install gymnasium[mujoco]

# 10. Run test (use nohup for long training)
nohup python airl_pipeline.py --env CartPole-v0 --demo-episodes 100 --airl-steps 2000000 > training.log 2>&1 &
```

**Subsequent runs:**

```bash
ssh -i your-key.pem ec2-user@your-instance-ip
cd ~/AIRL
source venv/bin/activate
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_IMITATION=0.0.0

# Run in background
nohup python airl_pipeline.py --env HalfCheetah-v1 --demo-episodes 100 --airl-steps 2000000 > training.log 2>&1 &
```


---
# Quick Start

### Basic Usage (Default Settings)

```bash
python airl_pipeline.py --env CartPole-v0
```

**What happens:**
1. Loads pre-trained CartPole expert from HuggingFace
2. Collects 60 expert demonstration episodes
3. Trains AIRL for 100,000 timesteps
4. Evaluates learner performance on 100 episodes

**Expected output:**
```
============================================================
AIRL Pipeline - CartPole-v0
============================================================
[1/4] Setting up environment: CartPole-v0
  ✓ Environment ready: seals:seals/CartPole-v0 (8 parallel envs)
[2/4] Loading expert policy from HuggingFace
  ✓ Expert loaded | Mean reward: 500.00 ± 0.00
[3/4] Collecting 60 expert demonstrations
  ✓ Collected 64 episodes, 32064 transitions
[4/4] Setting up AIRL trainer
  ✓ AIRL trainer initialized
  → Minimum training timesteps: 8192

Training AIRL (100000 steps)
============================================================
Mean reward before AIRL training:  100.35 ± 14.77
round: 100%|█████████████████████████████████████████████████████████████████████████████| 6/6 [01:14<00:00, 12.47s/it]

Learner reward after: 44.35 ± 2.52
Improvement: -56.00
============================================================

============================================================
Pipeline complete!
============================================================
```
'Note that this run results in an under-trained learner model and is just for demonstration.'

## CLI Parameters

### Environment & Basic Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--env` | `CartPole-v0` | Gym environment name (CartPole-v0) |
| `--seed` | `42` | Random seed for reproducibility |
| `--n-envs` | `8` | Number of parallel environments |

### Demonstration Collection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--demo-episodes` | `60` | Number of expert episodes to collect for training dataset |

**What it means:** More episodes = better representation of expert behavior, but diminishing returns after ~100 episodes.

### AIRL Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--airl-steps` | `100000` | Total training timesteps for AIRL |
| `--demo-batch-size` | `2048` | Batch size for demonstration samples |
| `--gen-buffer-capacity` | `512` | Generator replay buffer capacity |
| `--n-disc-updates` | `16` | Discriminator updates per training round |

**Minimum timesteps:** `gen-buffer-capacity × n-disc-updates` (default: 512 × 16 = 8,192)

### Evaluation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-eval-episodes` | `100` | Number of episodes for performance evaluation |

**What it means:** Used only for measuring reward before/after training. More episodes = more accurate estimate.

# Usage Examples

### Example 1: Quick Test Run

Test AIRL with minimal training:

```bash
python airl_pipeline.py --env CartPole-v0 --airl-steps 50000 --demo-episodes 30
```
```bash
python airl_pipeline.py --env HalfCheetah-v1 --airl-steps 50000 --demo-episodes 30
```

**Use case:** Fast prototyping, debugging, or verifying setup works.

### Example 2: Full Training to Match Expert

Train until the learner matches the expert performance:

```bash
python airl_pipeline.py --env CartPole-v0 --airl-steps 2500000 --demo-episodes 100
```

**Expected results:**
- Expert reward: ~500
- Learner before: ~40-100
- Learner after: ~450-500

```bash
python airl_pipeline.py --env HalfCheetah-v1 --airl-steps 2000000 --demo-episodes 100
```
**Expected results:**
- Expert reward: ~1630-1710
- Learner before: ~0
- Learner after: ~1000-1710


**Use case:** Production-quality imitation learning.

### Example 3: Custom AIRL Configuration 

Adjust AIRL hyperparameters:

```bash
python airl_pipeline.py \
    --seed 32
    --n-envs 10
    --env CartPole-v0 \
    --airl-steps 500000 \
    --demo-batch-size 1024 \
    --gen-buffer-capacity 2048 \
    --n-disc-updates 8
```

**Note:** Minimum timesteps becomes 2048 × 8 = 16,384

**Use case:** Hyperparameter tuning for specific environments.

### Example 4: More Parallel Environments [Not Tested]

Speed up training with more parallel environments:

```bash
python airl_pipeline.py --env CartPole-v0 --n-envs 16 --airl-steps 1000000
```

**Use case:** Faster training on multi-core systems.

# Understanding the Output

### Stage 1: Environment Setup
```
[1/4] Setting up environment: CartPole-v0
  ✓ Environment ready: seals:seals/CartPole-v0 (8 parallel envs)
```
- Creates vectorized environment for parallel execution
- Wraps with `RolloutInfoWrapper` for demonstration collection

### Stage 2: Expert Loading
```
[2/4] Loading expert policy from HuggingFace
  ✓ Expert loaded | Mean reward: 500.00 ± 0.00
```
- Loads pre-trained expert from HuggingFace Hub
- Expert reward shows the performance goal for AIRL

### Stage 3: Demonstration Collection
```
[3/4] Collecting 60 expert demonstrations
  ✓ Collected 67 episodes, 33500 transitions
```
- Collects expert trajectories (observations, actions, next_obs, done)
- May collect more episodes than requested to reach minimum timesteps
- Transitions = individual (state, action, next_state) tuples

### Stage 4: AIRL Setup
```
[4/4] Setting up AIRL trainer
  ✓ AIRL trainer initialized
  → Minimum training timesteps: 8192
```
- Initializes learner policy (untrained PPO)
- Creates a reward network for AIRL
- Shows minimum timesteps required (auto-calculated)

### Training & Evaluation
```
Training AIRL (100000 steps)
============================================================
Mean reward before AIRL training: 22.45 ± 8.12

Training for 100000 timesteps...

Mean reward after AIRL training: 487.23 ± 15.34
Improvement: +464.78
============================================================
```

**Interpreting results:**
- **Before training**: Random/untrained policy performance (~20-100 for CartPole)
- **After training**: Learned policy performance (should approach expert ~500)
- **Improvement**: Difference showing learning effectiveness
- **Standard deviation (±)**: Variance across evaluation episodes

**Success criteria:**
- ✅ After training ≈ Expert reward (within ~10%)
- ✅ Improvement > 80% of gap between before and expert
- ❌ After training << Expert reward → Need more training steps or demos

# Troubleshooting

### Error: Timesteps too low
```
AssertionError: No updates (need at least 16384 timesteps, have only total_timesteps=10000)!
```

**Solution:** Increase `--airl-steps` to at least the minimum shown in setup:
```bash
python airl_pipeline.py --env CartPole-v0 --airl-steps 20000
```

### Performance doesn't match the expert

**Possible causes:**
1. **Not enough training**: Increase `--airl-steps` (try 1-2M)
2. **Not enough demos**: Increase `--demo-episodes` (try 100+)
3. **Evaluation variance**: Increase `--n-eval-episodes` to 100+




# Advanced: Loading Custom Experts [Under Development]

For environments without pre-trained HuggingFace experts, use the separate loader:

```python
from load_cheetah_expert import train_cheetah_expert

# Train custom expert
expert, env = train_cheetah_expert(
    total_timesteps=1_000_000,
    save_path="my_expert.zip"
)

# Integrate with pipeline
pipeline = AIRLPipeline(env_name="HalfCheetah-v3")
pipeline.venv = env
pipeline.expert = expert
pipeline.collect_demonstrations()
pipeline.setup_airl()
pipeline.train_airl()
```

# Key Concepts

### Demo Episodes vs Eval Episodes

- **Demo Episodes** (`--demo-episodes`): Training data for AIRL
  - Used to teach the learner what expert behavior looks like
  - More = better training signal
  
- **Eval Episodes** (`--n-eval-episodes`): Testing/measurement only
  - Used to calculate performance metrics
  - More = more stable/accurate estimates
  - Does NOT affect training

### AIRL Hyperparameters

- **demo_batch_size**: How many expert transitions per batch
  - Larger = more stable discriminator training
  - Should be > 1024 for most environments

- **gen_replay_buffer_capacity**: Buffer size for learner experience
  - Larger = more diverse training data
  - Trade-off with memory usage

- **n_disc_updates**: Discriminator training frequency
  - Higher = better reward learning, but slower
  - Typical range: 8-32

## Citation

```bibtex
@misc{gleave2022imitation,
  author = {Gleave, Adam and Taufeeque, Mohammad and Rocamonde, Juan and Jenner, Erik and Wang, Steven H. and Toyer, Sam and Ernestus, Maximilian and Belrose, Nora and Emmons, Scott and Russell, Stuart},
  title = {imitation: Clean Imitation Learning Implementations},
  year = {2022},
  howPublished = {arXiv:2211.11972v1 [cs.LG]},
  archivePrefix = {arXiv},
  eprint = {2211.11972},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2211.11972},
}
```

## License

This implementation uses the [imitation library](https://github.com/HumanCompatibleAI/imitation) which is licensed under MIT.
