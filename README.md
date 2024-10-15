# Multi-object Fetch

Visual RL environments for multi-object reasoning and manipulation with a Fetch robot. 
The following table shows examples of the different environments (Reach, Push, and Pick) as well as the relational
reasoning tasks (Red, Reddest, Odd, OddGroups).

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
    <div style="width:6%; transform: rotate(-90deg); color: transparent">
        Tasks
    </div>
    <div style="width:22%; text-align: center">
        <p>Red</p>
    </div>
    <div style="width:22%; text-align: center">
        <p>Reddest</p>
    </div>
    <div style="width:22%; text-align: center">
        <p>Odd</p>
    </div>
    <div style="width:22%; text-align: center">
        <p>OddGroups</p>
    </div>
</div>

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
    <div style="width:6%; transform: rotate(-90deg); text-align: center">
        <p style="margin:0;">Reach</p>
    </div>
    <div style="width:22%;">
        <img src="docs/images/ReachRed.png">
    </div>
    <div style="width:22%;">
        <img src="docs/images/ReachReddest.png">
    </div>
    <div style="width:22%;">
        <img src="docs/images/ReachOdd.png">
    </div>
    <div style="width:22%;">
        <img src="docs/images/ReachOddGroups.png">
    </div>
</div>

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
     <div style="width:6%; transform: rotate(-90deg); text-align: center">
        <p>Push</p>
    </div>
    <div style="width:22%;">
        <img src="docs/images/PushRed.png">
    </div>
    <div style="width:22%;">
        <img src="docs/images/PushReddest.png">
    </div>
    <div style="width:22%;">
        <img src="docs/images/PushOdd.png">
    </div>
    <div style="width:22%;">
        <img src="docs/images/PushOddGroups.png">
    </div>
</div>

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
     <div style="width:6%; transform: rotate(-90deg); text-align: center">
        Pick
    </div>
    <div style="width:22%;">
        <img src="docs/images/PickRed.png">
    </div>
    <div style="width:22%;">
        <img src="docs/images/PickReddest.png">
    </div>
    <div style="width:22%;">
        <img src="docs/images/PickOdd.png">
    </div>
    <div style="width:22%;">
        <img src="docs/images/PickOddGroups.png">
    </div>
</div>

## Installation
The easiest way to install is to use the provided `create_conda_env.sh` script. This creates a conda environment called `mof` with all the necessary dependencies, sets up MuJoCo and copies asset files.
```
./create_conda_env.sh
source ~/.bashrc
conda activate mof
```

## Getting Started
To verify that the installation was successful, environments can be run with a random or user-controlled policy. An example command to control the robot on a pick-and-place task is:
``` 
python examples/run.py --policy user --task Odd --num_distractors 2 --environment Pick
```


The following arguments are used to configure the environments and can be tested in `examples/run.py`:
- `policy`: Whether to control the agent via the keyboard or run a random policy be in `random`, `user`.
- `environment`: Selects the environment to run and can be in `Reach`, `Push`, and `Pick`.
- `task`: Selects which task to run and can be in `Red`, `Reddest`, `Odd`, and `OddGroups`.
- `num_distractors`: Number of distractor targets/blocks to use in the environment.

## Credits
This repository is an extension of the environments in [fetch-block-construction](https://github.com/richardrl/fetch-block-construction).