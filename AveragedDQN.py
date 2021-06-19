import os
import gym
import json
import random
from datetime import datetime
import argparse
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the DQN hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 4        # how often to update the network


parser = argparse.ArgumentParser(description='Avg DQN')
parser.add_argument('--gpu_no', type=str, default='0', metavar='i-th', help='No GPU')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='Seed numb.')
parser.add_argument('--num_model', type=int, default=10, metavar='N', help='K in Avg DQN')
parser.add_argument('--env_name', type=str, default='LunarLander-v2', metavar='N', help='environment for experiment')
parser.add_argument('--n_episodes', type=int, default=2000, metavar='N', help='number of episode for training')
parser.add_argument('--quick_stop', type=str, default='n', help='train util last episode? [y/n]')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

args = parser.parse_args()

LR = args.lr

env_name = (args.env_name).replace('\r', '')
env = gym.make(env_name)
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

#GLOBAL VAR
last_episode = 0
last_score = 0

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
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
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_targets = deque()
        for _ in range(args.num_model):
            self.qnetwork_targets.append(QNetwork(state_size, action_size, seed).to(device))
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
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
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

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_targets, TAU)                     

    def soft_update(self, local_model, target_models, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        target_model = target_models[self.replace_model]
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            target_param.data.copy_(local_param.data)
        #update list maintaining the newest-oldest target model order: newest is in the last element
        self.qnetwork_targets_idx.remove(self.replace_model)
        self.qnetwork_targets_idx.append(self.replace_model)

        self.replace_model += 1
        if self.replace_model >= args.num_model:
            self.replace_model -= args.num_model



    def take_value_estimate(self, state):
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            value_estimate = 0
            for k in range(args.num_model-1,-1,-1): #loop from the newest to the oldest model, in case later needed for recency-weighting avg
                value_estimate += self.qnetwork_targets[k](state)
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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def dqn(agent, n_episodes, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    global last_score, last_episode
    scores = []                            # list containing scores from each episode
    scores_window = deque(maxlen=100)      # last 100 scores
    value_ests = []                        # list containing value estimates from each episode
    value_ests_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                        # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0; value_est = 0; n_steps = 0.0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            value_est += agent.take_value_estimate(state)
            state = next_state
            score += reward
            n_steps += 1.0
            if done:
                break
        value_est = (float)(value_est)/(float)(n_steps)
        scores_window.append(score); value_ests_window.append(value_est)
        scores.append(score)                                    # save most recent score
        value_ests.append(value_est)                            # save most recent value est.
        last_episode = i_episode; last_score = score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tValue Est.: {:.2f}'.format(i_episode,
                                                                               np.mean(scores_window),
                                                                               np.mean(value_ests_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tValue Est.: {:.2f}'.format(i_episode,
                                                                                   np.mean(scores_window),
                                                                                   np.mean(value_ests_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            if args.quick_stop == 'y':
                break
    return scores, value_ests

agent = Agent(env.observation_space.shape[0], env.action_space.n, seed=args.seed)
scores, value_ests = dqn(agent, args.n_episodes)

# create log file
time_now = datetime.now()
time_string = time_now.strftime("%Y-%m-%d_%H-%M-%S")
log = {'time': time_string, 'last_episode': last_episode, 'last_score': last_score, 'scores': scores, 'value_est': value_ests}

# save other parameters just in case that we change them in future experiments
log.update({'NETWORK': repr(agent.qnetwork_local),
            'BUFFER_SIZE': BUFFER_SIZE,
            'BATCH_SIZE': BATCH_SIZE,
            'GAMMA': GAMMA,
            'TAU': TAU,
            'LR': LR,
            'UPDATE_EVERY': UPDATE_EVERY
})

with open('log/'+env_name.lower()+'_k'+str(args.num_model)+"_seed"+str(args.seed)+".json", 'w') as outfile:
    json.dump(log, outfile)

# # plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.savefig('Averaged-DQN.png')
