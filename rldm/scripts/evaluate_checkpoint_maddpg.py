from __future__ import absolute_import, division, print_function
import argparse
import os
import warnings;
warnings.filterwarnings('ignore')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import collections
import os
import random
import tempfile
from argparse import RawTextHelpFormatter
import ray.rllib.contrib.maddpg.maddpg as maddpg
import gfootball.env as football_env
import gym
import numpy as np
import ray
import ray.cloudpickle as cloudpickle
import torch
from gfootball import env as fe
from gym import wrappers
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune.registry import get_trainable_cls, register_env
from ray.tune.schedulers import ASHAScheduler
from rldm.utils import football_tools_maddpg as ft
from rldm.utils import gif_tools as gt
from rldm.utils import system_tools as st
from gym.spaces import Dict, Discrete, Tuple, MultiDiscrete
import ray.rllib.agents.qmix.qmix as qmix
from ray.tune.registry import register_trainable, register_env
import tensorflow as tf
import time

EXAMPLE_USAGE = """
Example usage:

    Test a checkpoint for 10 episodes on the original scenario:
        python -m rldm.scripts.evaluate_checkpoint -c <path to checkpoint>
        NOTE: Checkpoints will look something like the following:
            "/mnt/logs/baselines/PPO_3_vs_3/checkpoint_009800/checkpoint-9800"

    Test a checkpoint for 10 episodes:
        python -m rldm.scripts.evaluate_checkpoint -c <path to checkpoint>

    Test a checkpoint for 100 episodes:
        python -m rldm.scripts.evaluate_checkpoint -c <path to checkpoint> -e 100

    NOTE:
        There are a few baseline checkpoints available for you to test:
          -> /mnt/logs/baselines/PPO_3_vs_3/checkpoint_009800/checkpoint-9800

        Test them:
          python -m rldm.scripts.evaluate_checkpoint -c /mnt/logs/train_agents_maddpg_self_3_vs_3_auto_GK_search_20000000_fifo_tune/contrib_MADDPG_3_vs_3_auto_GK_4fdc9_00000_0_actor_feature_reg=0.0005,actor_hidden_activation=relu,actor_hiddens=[128, 64],actor_lr_2021-11-25_00-12-01/checkpoint_009200/checkpoint-9200

        python -m rldm.scripts.evaluate_checkpoint_maddpg -c /mnt/logs/train_agents_maddpg_self_3_vs_3_auto_GK_search_20000000_fifo_tune/contrib_MADDPG_3_vs_3_auto_GK_4fdc9_00000_0_actor_feature_reg=0.0005,actor_hidden_activation=relu,actor_hiddens=[128,\ 64],actor_lr_2021-11-25_00-12-01/checkpoint_009200/checkpoint-9200 -e 5
        

"""


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


STAT_FUNC = {
    'min': np.min,
    'mean': np.mean,
    'median': np.median,
    'max': np.max,
    'std': np.std,
    'var': np.var,
}


def main(checkpoint, algorithm, env_name, config, num_episodes, debug):
    ray.init(log_to_driver=debug, include_dashboard=False,
             local_mode=True, logging_level='DEBUG' if debug else 'ERROR')
    print("Tensorflow GPU detected: ", tf.test.is_gpu_available())

    obs_space, act_space = ft.get_obs_act_space(env_name)

    grouping = {
        "group_1": [0, 1],
    }

    MADDPGAgent = maddpg.MADDPGTrainer.with_updates()
    register_trainable("MADDPG", MADDPGAgent)
    # cls = get_trainable_cls("MADDPG")
    obs_space_tuple = Tuple([obs_space['player_0'], obs_space['player_1']])
    act_space_tuple = Tuple([act_space['player_0'], act_space['player_1']])

    register_env(env_name, lambda _: ft.RllibGFootball(env_name=env_name).with_agent_groups(grouping, obs_space=obs_space_tuple, act_space=act_space_tuple))
    cls = get_trainable_cls(algorithm)

    agent = cls(env=env_name, config=config)
    agent.restore(checkpoint)

    env = ft.RllibGFootball(env_name=env_name).with_agent_groups(grouping, obs_space=obs_space_tuple, act_space=act_space_tuple)

    policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]
    policy_map = agent.workers.local_worker().policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    eps_stats = {}
    eps_stats['rewards_total'] = []
    eps_stats['timesteps'] = []
    eps_stats['score_reward'] = []
    eps_stats['win_perc'] = []
    eps_stats['undefeated_perc'] = []
    eps_stats['pass_attempts'] = []
    eps_stats['num_success_pass'] = []
    eps_stats['shoot_attempts'] = []

    wall_clock = []

    for eidx in range(num_episodes):

        wall_clock_episode = []

        mapping_cache = {}  # in case policy_agent_mapping is stochastic

        obs = env.reset()
        agent_states = state_init
        prev_actions = action_init
        prev_rewards = collections.defaultdict(lambda: 0.)

        done = False
        eps_stats['rewards_total'].append({a: 0.0 for a in obs})
        eps_stats['timesteps'].append(-1)
        eps_stats['score_reward'].append(0)
        eps_stats['win_perc'].append(0)
        eps_stats['undefeated_perc'].append(0)
        eps_stats['pass_attempts'].append(0)
        eps_stats['num_success_pass'].append(0)
        eps_stats['shoot_attempts'].append(0)

        while not done:
            actions = {}
            time_start = time.time()

            # observation_tuple = tuple([obs["group_1"][0], obs["group_1"][1]])
            for player_id, a_obs in obs.items():
                policy_id = mapping_cache.setdefault(
                    player_id, policy_agent_mapping(player_id, None))

                # print(player_id)
                p_use_lstm = use_lstm[policy_id]
                if p_use_lstm:
                    a_action, p_state, _ = agent.compute_single_action(
                        a_obs,
                        state=agent_states[player_id],
                        prev_action=prev_actions[player_id],
                        prev_reward=prev_rewards[player_id],
                        policy_id=policy_id)
                    agent_states[player_id] = p_state
                else:
                    a_action = agent.compute_single_action(
                        a_obs,
                        prev_action=prev_actions[player_id],
                        prev_reward=prev_rewards[player_id],
                        policy_id=policy_id)

                a_action = flatten_to_single_ndarray(a_action)
                actions[player_id] = a_action
                prev_actions[player_id] = a_action

            time_end = time.time()
            wall_clock_episode.append(time_end - time_start)

            next_obs, rewards, dones, infos = env.step(actions)

            # print(next_obs)

            done = dones['__all__']
            for player_id, r in rewards.items():
                prev_rewards[player_id] = r
                eps_stats['rewards_total'][-1][player_id] += r

            obs = next_obs
            eps_stats['timesteps'][-1] += 1

        wall_clock.append(wall_clock_episode)

        # print(eps_stats)
        # print(infos)
        eps_stats['score_reward'][-1] = infos['player_0']['score_reward']
        eps_stats['pass_attempts'][-1] = infos['player_0']["game_info"]["pass_attempts"] + infos['player_1']["game_info"]["pass_attempts"]
        eps_stats['shoot_attempts'][-1] = infos['player_0']["game_info"]["shoot_attempts"] + infos['player_1']["game_info"]["shoot_attempts"]
        eps_stats['num_success_pass'][-1] = infos['player_0']["game_info"]["num_success_pass"] + infos['player_1']["game_info"]["num_success_pass"]
        game_result = "loss" if infos['player_0']['score_reward'] == -1 else \
            "win" if infos['player_0']['score_reward'] == 1 else "tie"
        eps_stats['win_perc'][-1] = int(game_result == "win")
        eps_stats['undefeated_perc'][-1] = int(game_result != "loss")
        print(f"\nEpisode #{eidx + 1} ended with a {game_result}:")
        for p, r in eps_stats['rewards_total'][-1].items():
            print("\t{} got episode reward: {:.2f}".format(p, r))
        print("\tTotal reward: {:.2f}".format(sum(eps_stats['rewards_total'][-1].values())))

    eps_stats['rewards_total'] = {k: [dic[k] for dic in eps_stats['rewards_total']] \
                                  for k in eps_stats['rewards_total'][0]}

    mean_stats = []


    print("\n\nAll trials completed:")
    for stat_name, values in eps_stats.items():
        print(f"\t{stat_name}:")
        if type(values) is dict:
            for stat_name2, values2 in values.items():
                print(f"\t\t{stat_name2}:")
                for func_name, func in STAT_FUNC.items():
                    print("\t\t\t{}: {:.2f}".format(func_name, func(values2)))
        else:
            for func_name, func in STAT_FUNC.items():
                print("\t\t{}: {:.2f}".format(func_name, func(values)))
                if func_name == "mean":
                    # print(str(func(values))+"\n")
                    mean_stats.append(func(values))

    rewards_total = np.sum(list(eps_stats["rewards_total"].values()), axis=0)
    np.save("/mnt/rldm/scripts/maddpg_total_reward_per_eps.npy", rewards_total)
    print("\nSaved to npy", rewards_total)

    pass_attempts = eps_stats["pass_attempts"]
    np.save("/mnt/rldm/scripts/maddpg_pass_attempts.npy", pass_attempts)
    print("\nSaved to npy", pass_attempts)

    shoot_attempts = eps_stats["shoot_attempts"]
    np.save("/mnt/rldm/scripts/maddpg_shoot_attempts.npy", shoot_attempts)
    print("\nSaved to npy", shoot_attempts)

    num_success_pass = eps_stats["num_success_pass"]
    np.save("/mnt/rldm/scripts/maddpg_num_success_pass.npy", num_success_pass)
    print("\nSaved to npy", num_success_pass)

    mean_values = np.array(mean_stats)
    np.save("/mnt/rldm/scripts/maddpg_mean_values.npy", mean_values[[2,4,5,6]])
    print("\nSaved to npy", mean_values[[2,4,5,6]])

    ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for testing RLDM's P3 baseline agents",
        formatter_class=RawTextHelpFormatter,
        epilog=EXAMPLE_USAGE)
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='[REQUIRED] Checkpoint from which to roll out.')
    parser.add_argument('-a', '--algorithm', type=str, default='contrib/MADDPG',
                        help="The algorithm or model to train. This may refer to the name\
                            \nof a built-on algorithm (e.g. RLLib's DQN or PPO), or a\
                            \nuser-defined trainable function or class registered in the \
                            \ntune registry.\
                            \nDefault: contrib/MADDPG.")
    parser.add_argument('-e', '--num-episodes', default=10, type=int,
                        help='Number of episodes to test your agent(s).\
                            \nDefault: 10 episodes.')
    parser.add_argument('-r', '--dry-run', default=False, action='store_true',
                        help='Print the training plan, and exit.\
                            \nDefault: normal mode.')
    parser.add_argument('-d', '--debug', default=False, action='store_true',
                        help='Set full script to debug.\
                            \nDefault: "INFO" output mode.')
    args = parser.parse_args()

    assert args.checkpoint, "A checkpoint string is required"

    assert args.num_episodes <= 500, "Must rollout a maximum of 100 episodes"
    assert args.num_episodes >= 1, "Must rollout at least 1 episode"

    n_cpus, _ = st.get_cpu_gpu_count()
    assert n_cpus >= 2, "Didn't find enough cpus"

    config = {}
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "../params.pkl")
    assert os.path.exists(config_path), "Could not find a proper config from checkpoint"
    with open(config_path, "rb") as f:
        config = cloudpickle.load(f)

    config["create_env_on_driver"] = True
    config["log_level"] = 'DEBUG' if args.debug else 'ERROR'
    config["num_workers"] = 1
    config["num_gpus"] = 0
    config["explore"] = False
    test_env_name = "3_vs_3_auto_GK"  # this semester we're only doing 3v3 w auto GK

    print("")
    print("Calling training code with following parameters (arguments affecting each):")
    print("\tLoading checkpoint (-c):", args.checkpoint)
    print("\tLoading driver (-a):", args.algorithm)
    print("\tEnvironment to load for testing (-l):", test_env_name)
    print("\tNumber of episodes to run (-e):", args.num_episodes)
    print("\tIs this a dry-run only (-r):", args.dry_run)
    print("\tScript running on debug mode (-d):", args.debug)
    print("")

    if not args.dry_run:
        main(args.checkpoint, args.algorithm, test_env_name, config, args.num_episodes, args.debug)
