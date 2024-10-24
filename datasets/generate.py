import gym
import multi_object_fetch
from multi_object_fetch.utils.parser import MOFParser
import numpy as np
import os.path
from PIL import Image
import shutil
import tqdm


def save_episode(path: str, env: gym.Env, width: int, height: int, action_repeat: int = 1) -> None:
    def save_image():
        Image.fromarray(env.render(mode='rgb_array', size=(width, height))).save(os.path.join(path, f'{step_count}.png'))

    os.mkdir(path)
    step_count, actions, rewards = 0, [], []
    _, done = env.reset(), False
    save_image()

    while not done:
        action = env.action_space.sample()
        for _ in range(action_repeat):
            if done:
                break
            _, reward, done, _ = env.step(action)

        step_count += 1
        save_image()
        actions.append(action)
        rewards.append(reward)

    np.save(os.path.join(path, 'actions.npy'), actions)
    np.save(os.path.join(path, 'rewards.npy'), rewards)


if __name__ == "__main__":
    parser = MOFParser()
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Directory in which to save the dataset.")
    parser.add_argument("--num_train", type=int, default=20000, help="The number of training videos to generate.")
    parser.add_argument("--num_val", type=int, default=2000, help="The number of validation videos to generate.")
    parser.add_argument("--num_test", type=int, default=2000, help="The number of test videos to generate.")
    parser.add_argument("--width", type=int, default=64, help="The width of the images.")
    parser.add_argument("--height", type=int, default=64, help="The height of the images.")
    parser.add_argument("--action_repeat", type=int, default=2, help="The number of times to repeat each action.")

    args = parser.parse_args()

    if os.path.exists(args.dataset_dir):
        shutil.rmtree(args.dataset_dir)
    os.mkdir(args.dataset_dir)

    min_distractors, max_distractors = 0, 4
    if args.task.startswith("Odd"):
        min_distractors = 2
        if args.task == "OddGroups":
            min_distractors = 4

    for num_distractors in range(min_distractors, max_distractors + 1):
        env = gym.make(f'{args.environment}{args.task}_{num_distractors}Distractors_{args.reward_type}-v1')

        for split in ["train", "val", "test"]:
            num_episodes = getattr(args, f"num_{split}") // (max_distractors - min_distractors + 1)
            if num_distractors == min_distractors:
                os.mkdir(os.path.join(args.dataset_dir, split))
            for episode in tqdm.tqdm(range(num_episodes), desc=split.capitalize(), postfix={"#Distractors": num_distractors}):
                save_episode(os.path.join(args.dataset_dir, split, str(episode * (max_distractors - min_distractors + 1) + (num_distractors - min_distractors))), env, width=args.width,
                             height=args.height, action_repeat=args.action_repeat)
