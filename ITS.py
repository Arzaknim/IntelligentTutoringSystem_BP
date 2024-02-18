import math
import random

import numpy as np
import torch
from torch import optim, nn
from MainEnvironment import MainEnvironment
from PeriodicTable import PeriodicTable
from Qnet import DQN
from ReplayMemory import ReplayMemory, Transition
from Student import Student
import matplotlib.pyplot as plt
import fire
import os
from datetime import date, datetime

from training_object import TrainingObject

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100000  # 16000
TAU = 0.005
LR = 1e-4
steps_done = 0
eps = None


def select_action(state, policy_net, env):
    global steps_done
    global eps
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    eps = eps_threshold
    print(eps)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def optimize_model(training_object):
    if len(training_object.memory) < BATCH_SIZE:
        return
    transitions = training_object.memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = training_object.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = training_object.target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    training_object.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(training_object.policy_net.parameters(), 100)
    training_object.optimizer.step()


def run(train):
    pt = PeriodicTable()
    block_strength = [1, 0.7, 1, 1]
    student = Student(0.7, 0.03, 1, block_strength)
    env = MainEnvironment(pt, student, 40)

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    valid_file = os.path.exists('qnet_w.pth')
    if not valid_file:
        target_net.load_state_dict(policy_net.state_dict())
    else:
        target_net.load_state_dict(torch.load('qnet_w.pth'))
        policy_net.load_state_dict(torch.load('qnet_w.pth'))

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    training_object = TrainingObject(memory, target_net, policy_net, optimizer)

    if torch.cuda.is_available():
        num_episodes = 100  # 2000
    else:
        num_episodes = 50
    ts_list = []
    score_list = []
    eps_list = []
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        score = 0
        while state is not None:
            action = select_action(state, policy_net, env)
            observation, reward, terminated, ts = env.step(action.item())
            score += reward
            reward = torch.tensor([reward], device=device)
            done = terminated

            if terminated:
                next_state = None
                ts_list.append(ts)
                score_list.append(score)
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(training_object)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                        1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

        print(f'Episode {i_episode}: {score}')
    time = str(datetime.now()).split()
    time[1] = time[1].split('.')[0].replace(':', '-')
    time = time[0] + '_' + time[1]
    print(f'qnet_w_{time}.pth')
    if train:
        torch.save(target_net.state_dict(), f'training_files/qnet_w_{time}.pth')

    mean_ts = []
    mean_score = []
    mean_from = 50
    for i in range(0, num_episodes, mean_from):
        temp_ts = 0
        temp_score = 0
        for j in range(mean_from):
            temp_ts += ts_list[i + j]
            temp_score += score_list[i + j]
        mean_ts.append(temp_ts / mean_from)
        mean_score.append(temp_score / mean_from)

    x_axis = np.linspace(1, num_episodes, num_episodes)
    x_axis_mean = np.linspace(1, num_episodes, int(num_episodes/mean_from))
    plt.title('Timestep graph')
    plt.plot(x_axis, np.array(ts_list))
    plt.plot(x_axis_mean, mean_ts, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Timestep')
    plt.savefig(f'training_files/timestep_{time}.jpg')
    plt.title('Score graph')
    plt.plot(x_axis, np.array(score_list))
    plt.plot(x_axis_mean, mean_score, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.savefig(f'training_files/score_{time}.jpg')


if __name__ == '__main__':
    # fire.Fire(run)
    run(True)
