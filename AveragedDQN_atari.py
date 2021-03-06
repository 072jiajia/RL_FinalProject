import os
import gym
import json
import random
from datetime import datetime
from itertools import count
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
BUFFER_SIZE = (int)(1e6)         # replay buffer size: paper uses 1M
BATCH_SIZE = 32                  # minibatch size
GAMMA = 0.99                     # discount factor
UPDATE_EVERY = 4                 # how often to update the network
FREEZE_INTERVAL = 10000          # the paper uses 10k
LAST_STEP_DECREASING_EPS = 1e6   # epsilon will decrease from 1 to 0.1 until this step
N_STEP_EVAL = (int)(1e4)         # agent will be evaluated per this number of training steps, & the paper uses 1M or 2M


parser = argparse.ArgumentParser(description='Avg DQN')
parser.add_argument('--gpu_no', type=str, default='5', metavar='i-th', help='No GPU')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='Seed numb.')
parser.add_argument('--num_model', type=int, default=10, metavar='N', help='K in Avg DQN')
parser.add_argument('--env_name', type=str, default='Breakout-v0', metavar='N', help='environment for experiment')
parser.add_argument('--total_frames', type=float, default=120e6, metavar='N', help='number of frames for training')
parser.add_argument('--quick_stop', type=str, default='n', help='train util last episode? [y/n]')
parser.add_argument('--lr', type=float, default=25e-5, help='learning rate') #paper uses 25e-5
parser.add_argument('--num_frame', type=int, default=4, help='use N latest frame to feed to the NN')

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
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Use GPU is possible else use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# GLOBAL VAR
LAST_EPISODE = 0
LAST_SCORE = 0
LAST_FRAMES = 0


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(args.num_frame, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_size),
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.conv(state)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x


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
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_targets = deque()
        for _ in range(args.num_model):
            self.qnetwork_targets.append(QNetwork(state_size, action_size).to(device))
        # create this list to maintain the order of newest to oldest target network
        self.qnetwork_targets_idx = [i for i in range(args.num_model)]
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
            self.update_target(self.qnetwork_local, self.qnetwork_targets)


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
            Q_targets_next = 0
            for qnetwork_target in self.qnetwork_targets:
                Q_targets_next += qnetwork_target(next_states)
            Q_targets_next = Q_targets_next / args.num_model
            Q_targets_next = Q_targets_next.detach().max(1)[0].unsqueeze(1)

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

    def update_target(self, local_model, target_models):
        """ Update model parameters """
        target_model = target_models[self.replace_model]
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

        #update list maintaining the newest-oldest target model order: newest is in the last element
        self.qnetwork_targets_idx.remove(self.replace_model)
        self.qnetwork_targets_idx.append(self.replace_model)

        self.replace_model += 1
        if self.replace_model >= args.num_model:
            self.replace_model -= args.num_model


    def take_value_estimate(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            value_estimate = 0
            for k in range(args.num_model-1,-1,-1): #loop from the newest to the oldest model, in case later needed for recency-weighting avg
                value_estimate += self.qnetwork_targets[self.qnetwork_targets_idx[k]](state)
            value_estimate = value_estimate / args.num_model
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

def rgb2gray(state):
    # some ref, e.g, https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    # use this weights to do rgb2gray
    return np.dot(state[..., :3], [0.299, 0.587, 0.114])

def evaluate(agent, env_name):
    state = env_test.reset()
    state = rgb2gray(state)
    state = cv2.resize(state, dsize=(84, 84))

    state = [state] * args.num_frame
    score = 0; value_est = 0; n_steps_cur_episode = 0.0
    for t in count():
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = rgb2gray(next_state)
        next_state = cv2.resize(next_state, dsize=(84, 84))
        next_state = state[1:] + [next_state]
        value_est += agent.take_value_estimate(state)
        state = next_state
        score += reward
        n_steps_cur_episode += 1.0
        if done:
            break
    return score, value_est/n_steps_cur_episode


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

    with open('log/recent' + str(args.num_frame) + '_' + env_name.lower()+'_k'+str(args.num_model)+"_seed"+str(args.seed)+".json", 'w') as outfile:
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

    for i_episode in count():
        i_episode += 1
        state = env.reset()
        state = rgb2gray(state)
        state = cv2.resize(state, dsize=(84, 84))

        state = [state] * args.num_frame
        score = 0; value_est = 0; n_steps_cur_episode = 0.0

        for t in count():
            action = agent.act(state, eps)
            current_eps = ((eps_end - eps_start) * (float)(total_steps) / LAST_STEP_DECREASING_EPS) + eps_start
            eps = max(eps_end, current_eps)  # use linear interpolation decay
            next_state, reward, done, _ = env.step(action)
            next_state = rgb2gray(next_state)
            next_state = cv2.resize(next_state, dsize=(84, 84))
            next_state = state[1:] + [next_state] #keep update for last args.num_frame
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
        print('\rEnv {}  Episode {}  Steps {}    '.format(env_name, i_episode, total_steps), end="")
        if i_episode % 100 == 0:
            print('\rEnv {}  Episode {}  Steps {}  Average Score: {:.2f}  Value Est.: {:.2f}'.format(env_name, i_episode,
                                                                                                     total_steps, score,
                                                                                                     value_est))
        if (total_steps >= total_frames): #quite training after total_frames
            break

        store_json(scores, value_ests)
    return scores, value_ests

state_size = env.observation_space.shape
agent = Agent(state_size=state_size,
              action_size=env.action_space.n,
              seed=args.seed,
              env_name=env_name)
scores, value_ests = dqn(agent, total_frames=args.total_frames)

