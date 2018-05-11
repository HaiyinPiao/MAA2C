import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import pickle

from MAA2C import MAA2C
from common.utils import agg_double_list

import sys
import matplotlib.pyplot as plt


MAX_EPISODES = 5000
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 10
EVAL_INTERVAL = 100

# roll out n steps
ROLL_OUT_N_STEPS = 10
# only remember the latest ROLL_OUT_N_STEPS
MEMORY_CAPACITY = ROLL_OUT_N_STEPS
# only use the latest ROLL_OUT_N_STEPS for training A2C
BATCH_SIZE = ROLL_OUT_N_STEPS

REWARD_DISCOUNTED_GAMMA = 0.99
ENTROPY_REG = 0.00
#
DONE_PENALTY = -10.

CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

RANDOM_SEED = 2017

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env):
    trainers = []
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    act_shape_n = [env.action_space[i].shape for i in range(env.n)]

    for i in range(env.n):
        trainers.append(MAA2C(env, env.n, obs_shape_n, act_shape_n))
    # for i in range(num_adversaries, env.n):
    #     trainers.append(trainer(
    #         "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
    #         local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def run(arglist):
    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    # Create agent trainers
    # obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries)

    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    act_shape_n = [env.action_space[i].n for i in range(env.n)]

    print(obs_shape_n)
    print(act_shape_n)
    a2c = MAA2C(env, env.n, obs_shape_n, act_shape_n)
    # trainers = get_trainers(env)

    # env = gym.make(env_id)
    # env.seed(RANDOM_SEED)
    # env_eval = gym.make(env_id)
    # env_eval.seed(RANDOM_SEED)
    # state_dim = env.observation_space.shape[0]
    # if len(env.action_space.shape) > 1:
    #     action_dim = env.action_space.shape[0]
    # else:
    #     action_dim = env.action_space.n

    # a2c = A2C(env=env, memory_capacity=MEMORY_CAPACITY,
    #           state_dim=state_dim, action_dim=action_dim,
    #           batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
    #           done_penalty=DONE_PENALTY, roll_out_n_steps=ROLL_OUT_N_STEPS,
    #           reward_gamma=REWARD_DISCOUNTED_GAMMA,
    #           epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
    #           epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
    #           episodes_before_train=EPISODES_BEFORE_TRAIN,
    #           critic_loss=CRITIC_LOSS)

    episodes =[]
    eval_rewards =[]
    while a2c.n_episodes < MAX_EPISODES:
        a2c.interact()
        # if a2c.n_episodes >= EPISODES_BEFORE_TRAIN:
        #     a2c.train()
        # if a2c.episode_done and ((a2c.n_episodes+1)%EVAL_INTERVAL == 0):
        #     rewards, _ = a2c.evaluation(env_eval, EVAL_EPISODES)
        #     rewards_mu, rewards_std = agg_double_list(rewards)
        #     print("Episode %d, Average Reward %.2f" % (a2c.n_episodes+1, rewards_mu))
        #     episodes.append(a2c.n_episodes+1)
        #     eval_rewards.append(rewards_mu)

    # episodes = np.array(episodes)
    # eval_rewards = np.array(eval_rewards)
    # np.savetxt("./output/%s_a2c_episodes.txt"%env_id, episodes)
    # np.savetxt("./output/%s_a2c_eval_rewards.txt"%env_id, eval_rewards)

    # plt.figure()
    # plt.plot(episodes, eval_rewards)
    # plt.title("%s"%env_id)
    # plt.xlabel("Episode")
    # plt.ylabel("Average Reward")
    # plt.legend(["A2C"])
    # plt.savefig("./output/%s_a2c.png"%env_id)


def train(arglist):
    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    # Create agent trainers
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries)
    trainers = get_trainers(env)
 
    # Initialize
    U.initialize()

    # Load previous results, if necessary
    if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
    if arglist.display or arglist.restore or arglist.benchmark:
        print('Loading previous state...')
        U.load_state(arglist.load_dir)

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    saver = tf.train.Saver()
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    while True:
        # get action
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)
        # collect experience
        for i, agent in enumerate(trainers):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        # increment global step counter
        train_step += 1

        # for benchmarking learned policies
        if arglist.benchmark:
            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])
            if train_step > arglist.benchmark_iters and (done or terminal):
                file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                print('Finished benchmarking, now saving...')
                with open(file_name, 'wb') as fp:
                    pickle.dump(agent_info[:-1], fp)
                break
            continue

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # update all trainers, if not in display or benchmark mode
        loss = None
        for agent in trainers:
            agent.preupdate()
        for agent in trainers:
            loss = agent.update(trainers, train_step)

        # save model, display training output
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            U.save_state(arglist.save_dir, saver=saver)
            # print statement depends on whether or not there are adversaries
            if num_adversaries == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
            else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > arglist.num_episodes:
            rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ep_rewards, fp)
            agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break

if __name__ == '__main__':
    arglist = parse_args()
    run(arglist)