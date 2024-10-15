import logging
from gym.envs.registration import register


logger = logging.getLogger(__name__)


for environment in ['Reach', 'Push', 'Pick']:
    for task in ['Red', 'Reddest', 'Odd', 'OddGroups']:
        for num_distractors in range(0, 5):
            for reward_type in ['sparse', 'dense']:
                if task.startswith('Odd') and num_distractors < 2:  # Odd-one-out needs at least two others.
                    continue
                if task == 'OddGroups' and num_distractors < 4:  # Odd with groups needs at least 2 x 2 others.
                    continue

                initial_qpos = {
                    'robot0:slide0': 0.405,
                    'robot0:slide1': 0.48,
                    'robot0:slide2': 0.0,
                }

                kwargs = {
                    'reward_type': reward_type,
                    'initial_qpos': initial_qpos,
                    'num_distractors': num_distractors,
                    'task': task
                }

                if environment != 'Reach':
                    for i in range(num_distractors + 1):
                        initial_qpos[F"object{i}:joint"] = [1.25, 0.53, .4 + i*.06, 1., 0., 0., 0.]

                    if environment == 'Push':
                        kwargs['target_in_the_air'] = False
                        kwargs['block_gripper'] = True



                register(
                    id=f'{environment}{task}_{num_distractors}Distractors_{reward_type.capitalize()}-v1',
                    entry_point='multi_object_fetch.env:ReachEnv' if environment == 'Reach' else 'multi_object_fetch.env:ManipulateEnv',
                    kwargs=kwargs,
                    max_episode_steps=50 if environment == 'Reach' else 100,
                )
