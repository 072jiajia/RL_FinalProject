import os
import gym
import json
import random
from datetime import datetime
import argparse
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the DQN hyperparameters
BUFFER_SIZE = (int)(1e4)         # replay buffer size: paper uses 1M
BATCH_SIZE = 32                  # minibatch size
GAMMA = 0.99                     # discount factor
UPDATE_EVERY = 4                 # how often to update the network
FREEZE_INTERVAL = 100          # the paper uses 10k
LAST_STEP_DECREASING_EPS = 1e6   # epsilon will decrease from 1 to 0.1 until this step
N_STEP_EVAL = (int)(100)         # agent will be evaluated per this number of training steps, & the paper uses 1M or 2M


parser = argparse.ArgumentParser(description='Avg DQN')
parser.add_argument('--gpu_no', type=str, default='5', metavar='i-th', help='No GPU')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='Seed numb.')
parser.add_argument('--env_name', type=str, default='Roulette-v0', metavar='N', help='environment for experiment')
parser.add_argument('--total_frames', type=float, default=120e6, metavar='N', help='number of frames for training')
parser.add_argument('--lr', type=float, default=25e-5, help='learning rate') #paper uses 25e-5

args = parser.parse_args()
LR = args.lr

env_name = (args.env_name).replace('\r', '')
env = gym.make(env_name)
env_test = gym.make(env_name)
print('State shape: ', env.observation_space.shape)
print('Number of Actions: ', env.action_space.n)

env.seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# device = torch.device("cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# GLOBAL VAR
LAST_EPISODE = 0
LAST_SCORE = 0
LAST_FRAMES = 0


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.Q = nn.Parameter(torch.rand(1, action_size), requires_grad=True)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # return self.Q + state
        return self.Q.repeat(state.shape[0], 1)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, env_name):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = QNetwork(action_size).to(device)
        self.qnetwork_target = QNetwork(action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.replace_model = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
        if self.t_step % FREEZE_INTERVAL == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model

        with torch.no_grad():
            _, next_state_actions = self.qnetwork_local(states).max(1, keepdim=True)
            Q_targets_next = self.qnetwork_target(states).gather(1, next_state_actions)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) #shape = (batch_size,1)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions) #shape = (batch_size,1)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)


    def take_value_estimate(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            value_estimate = 0
            value_estimate += self.qnetwork_target(state)
            value_estimate = np.max(value_estimate.cpu().detach().numpy())
        return value_estimate


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.FloatTensor(np.stack([e.state for e in experiences if e is not None])).to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.FloatTensor(np.vstack([e.reward for e in experiences if e is not None])).to(device)
        next_states = torch.FloatTensor(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def evaluate(agent, env_name):
    state = env_test.reset()
    state = [state]

    score = 0; value_est = 0; n_steps_cur_episode = 0.0
    from itertools import count
    for t in count():
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = [next_state]
        value_est += agent.take_value_estimate(state)
        state = next_state
        score += reward
        n_steps_cur_episode += 1.0
        if done:
            break
    return score/n_steps_cur_episode, value_est/n_steps_cur_episode


def store_json(scores, value_ests):
    # create log file
    time_now = datetime.now()
    time_string = time_now.strftime("%Y-%m-%d_%H-%M-%S")
    log = {'time': time_string, 'last_steps': LAST_FRAMES, 'last_episode': LAST_EPISODE, 'last_score': LAST_SCORE, 'scores': scores, 'value_est': value_ests}

    # save other parameters just in case that we change them in future experiments
    log.update({'NETWORK': repr(agent.qnetwork_local),
                'BUFFER_SIZE': BUFFER_SIZE,
                'BATCH_SIZE': BATCH_SIZE,
                'GAMMA': GAMMA,
                'LR': LR,
                'UPDATE_EVERY': UPDATE_EVERY,
                'FREEZE_INTERVAL': FREEZE_INTERVAL,
                'LAST_STEP_DECREASING_EPS': LAST_STEP_DECREASING_EPS,
                'N_STEP_EVAL': N_STEP_EVAL
    })

    with open('log/' + env_name.lower()+'_ddqn_seed'+str(args.seed)+".json", 'w') as outfile:
        json.dump(log, outfile)


def dqn(agent, total_frames, max_t=1000, eps_start=1.0, eps_end=0.1, eps_decay=0.995): #paper use eps_end = 0.1
    """Deep Q-Learning.
    
    Params
    ======
        total_frames (float): maximum number of training frames
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    global LAST_SCORE, LAST_EPISODE, LAST_FRAMES
    scores = []                            # list containing scores from each episode
    scores_window = deque(maxlen=100)      # last 100 scores
    value_ests = []                        # list containing value estimates from each episode
    value_ests_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                        # initialize epsilon
    total_steps = 0                        # initialize total frames
    from itertools import count as cnt1
    # for i_episode in range(1, n_episodes+1):
    for i_episode in cnt1():
        i_episode += 1
        state = env.reset()
        state = [state]

        score = 0; value_est = 0; n_steps_cur_episode = 0.0

        # for t in range(max_t):
        from itertools import count as cnt2
        for t in cnt2():
            action = agent.act(state, eps)
            current_eps = ((eps_end - eps_start) * (float)(total_steps) / LAST_STEP_DECREASING_EPS) + eps_start
            eps = max(eps_end, current_eps)  # use linear interpolation decay
            next_state, reward, done, _ = env.step(action)
            next_state = [next_state]
            agent.step(state, action, reward, next_state, done)

            if (total_steps%N_STEP_EVAL == 0): #evaluate the agent per N_STEP_EVAL
                score, value_est = evaluate(agent,args.env_name)
                scores_window.append(score)
                value_ests_window.append(value_est)
                scores.append(score)  # save most recent score
                value_ests.append(value_est)  # save most recent value est.
                LAST_SCORE = score
            state = next_state
            n_steps_cur_episode += 1.0
            total_steps += 1
            LAST_FRAMES = total_steps
            if done:
                break

        LAST_EPISODE = i_episode
        print('\rEnv {}    Episode {}    Steps {}    Average Score: {:.2f}    Value Est.: {:.2f}'.format(env_name, i_episode,
                                                                                                 total_steps, score,
                                                                                                 value_est), end="")
        if i_episode % 100 == 0:
            store_json(scores, value_ests)
            print('\rEnv {}    Episode {}    Steps {}    Average Score: {:.2f}    Value Est.: {:.2f}'.format(env_name, i_episode,
                                                                                                     total_steps, score,
                                                                                                     value_est))

        if (total_steps >= total_frames): #quite training after total_frames
            break

    return scores, value_ests

state_size = env.observation_space.shape
agent = Agent(state_size=state_size,
              action_size=env.action_space.n,
              seed=args.seed,
              env_name=env_name)
scores, value_ests = dqn(agent, total_frames=args.total_frames)

