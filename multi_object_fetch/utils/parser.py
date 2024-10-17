import argparse


class MOFParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("--environment", type=str, choices=["Reach", "Push", "Pick"], default="Reach")
        self.add_argument("--task", type=str, choices=['Red', 'Reddest', 'Odd', 'OddGroups'], default="Red",
                          help="The task to run.")
        self.add_argument("--num_distractors", type=int, default=2,
                          help="The number of distractors in the environment. Must be between 0 and 10.")
        self.add_argument("--reward_type", type=str, choices=["Sparse", "Dense"], default="Dense",)
