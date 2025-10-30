[![CircleCI](https://circleci.com/gh/HumanCompatibleAI/imitation.svg?style=svg)](https://circleci.com/gh/HumanCompatibleAI/imitation)
[![Documentation Status](https://readthedocs.org/projects/imitation/badge/?version=latest)](https://imitation.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/imitation.svg)](https://badge.fury.io/py/imitation)

# Adversarial Inverse Reinforcement Learning Baseline Implementations

This project aims to provide clean implementations of the Adversarial Inverse Reinforcement Learning (AIRL) algorithm.
The algorithm supports discrete action spaces.


| [Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248)                                                    | [`algoritms.airl`](https://imitation.readthedocs.io/en/latest/algorithms/airl.html)                                      | 

You can find the original [documentation here](https://imitation.readthedocs.io/en/latest/).

## Installation

### Prerequisites

- Python 3.8+
- (Optional) OpenGL (to render Gymnasium environments)
- (Optional) FFmpeg (to encode videos of renders)

> Note: `imitation` is only compatible with newer [gymnasium](https://gymnasium.farama.org/) environment API and does not support the older `gym` API.

### Installing PyPI release

Installing the PyPI release is the standard way to use `imitation`, and the recommended way for most users.

```
pip install imitation
```

### Install from source

If you like, you can install `imitation` from source to [contribute to the project][contributing] or access the very last features before a stable release. You can do this by cloning the GitHub repository and running the installer directly. First run:
`git clone http://github.com/HumanCompatibleAI/imitation && cd imitation`.

For development mode, then run:

```
pip install -e ".[dev]"
```

This will run `setup.py` in development mode, and install the additional dependencies required for development. For regular use, run instead

```
pip install .
```

Additional extras are available depending on your needs. Namely, `tests` for running the test suite, `docs` for building the documentation, `parallel` for parallelizing the training, and `atari` for including atari environments. The `dev` extra already installs the `tests`, `docs`, and `atari` dependencies automatically, and `tests` installs the `atari` dependencies.

For macOS users, some packages are required to run experiments (see `./experiments/README.md` for details). First, install Homebrew if not available (see [Homebrew](https://brew.sh/)). Then, run:

```
brew install coreutils gnu-getopt parallel
```

## CLI Quickstart

We provide several CLI scripts as a front-end to the algorithms implemented in `imitation`. These use [Sacred](https://github.com/idsia/sacred) for configuration and replicability.

From [examples/quickstart.sh:](examples/quickstart.sh)

```bash
# Train PPO agent on pendulum and collect expert demonstrations. Tensorboard logs saved in quickstart/rl/
python -m imitation.scripts.train_rl with pendulum environment.fast policy_evaluation.fast rl.fast fast logging.log_dir=quickstart/rl/

# Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial gail with pendulum environment.fast demonstrations.fast policy_evaluation.fast rl.fast fast demonstrations.path=quickstart/rl/rollouts/final.npz demonstrations.source=local

# Train AIRL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial airl with pendulum environment.fast demonstrations.fast policy_evaluation.fast rl.fast fast demonstrations.path=quickstart/rl/rollouts/final.npz demonstrations.source=local
```

Tips:

- Remove the "fast" options from the commands above to allow training run to completion.
- `python -m imitation.scripts.train_rl print_config` will list Sacred script options. These configuration options are documented in each script's docstrings.

For more information on how to configure Sacred CLI options, see the [Sacred docs](https://sacred.readthedocs.io/en/stable/).

## Python Interface Quickstart

See [examples/quickstart.py](examples/quickstart.py) for an example script that loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.


# Citations (BibTeX)

```
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

# Contributing

See [Contributing to imitation][contributing] for more information.


[contributing]: https://imitation.readthedocs.io/en/latest/development/contributing/index.html
