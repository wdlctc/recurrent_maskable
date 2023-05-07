# RecurrentMaskablePPO

RecurrentMaskablePPO is a custom implementation of the Proximal Policy Optimization (PPO) algorithm, designed specifically for environments with recurrent states and maskable actions. This implementation is based on the [stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) repository, which extends the popular reinforcement learning library, [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

## Features

- Compatible with environments that have recurrent states and require masking of certain actions.
- Built on top of the stable-baselines3 library, inheriting its modularity and ease of use.
- Efficient and scalable implementation for complex tasks.

## Installation

To install RecurrentMaskablePPO, follow the steps below:

1. Make sure you have Python 3.7 or later installed on your system. You can download the latest version from the [official Python website](https://www.python.org/downloads/).

2. Install stable-baselines3 and stable-baselines3-contrib:

```bash
pip install stable-baselines3 stable-baselines3-contrib
```

3. Clone this repository:

```bash
git clone https://github.com/yourusername/RecurrentMaskablePPO.git
```

4. Navigate to the cloned repository and install the package:

```bash
cd recurrent_msakable
pip install -e .
```
