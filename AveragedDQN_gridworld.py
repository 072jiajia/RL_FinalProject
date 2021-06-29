import os
import json
import random
from datetime import datetime
import argparse
import numpy as np
import time
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gridworld_env import Env

# Define the DQN hyperparameters
BUFFER_SIZE = 10000             # replay buffer size: paper uses 1M
BATCH_SIZE = 32                  # mini-batch size
GAMMA = 0.9                      # discount factor
UPDATE_EVERY = 4                 # how often to update the network
FREEZE_INTERVAL = 10000          # the paper uses 10k
LAST_STEP_DECREASING_EPS = 10000 # epsilon will decrease from 1 to 0.1 until this step
N_STEP_EVAL = (int)(1e4)         # agent will be evaluated per this number of training steps, & the paper uses 1M or 2M


parser = argparse.ArgumentParser(description='Avg DQN')
parser.add_argument('--gpu_no', type=str, default='5', metavar='i-th', help='No GPU')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='Seed numb.')
parser.add_argument('--num_model', type=int, default=5, metavar='N', help='K in Avg DQN')
parser.add_argument('--env_name', type=str, default='GridWorld', metavar='N', help='environment for experiment')
parser.add_argument('--total_steps', type=float, default=2e6, metavar='N', help='number of steps for training')
parser.add_argument('--quick_stop', type=str, default='n', help='train util last episode? [y/n]')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--grid_size', type=int, default=20, help='size of the gridworld')
parser.add_argument('--step_size_testing', type=int, default=10000, help='step size of testing')


args = parser.parse_args()
LR = args.lr

env_name = (args.env_name).replace('\r', '')
env = Env(grid=(args.grid_size,args.grid_size), seed=args.seed)
env_test = Env(grid=(args.grid_size,args.grid_size), seed=args.seed)
print('State shape: ', env.gridsize)
print('Number of Actions: ', env.action_space.shape)

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


def init_buffer(env, agent):
    '''
        initialize all possible transitions into buffer
    '''
    n_states = env.gridsize

    # there are row*col-1 possibible agent location
    for row in range(n_states[0]):
        for col in range(n_states[1]):
            for action in range(4):
                if row == n_states[0]-1 and col == n_states[1]-1:
                    break

                # action(up:0, down:1, right:2, left:3)
                state = env.set_agent_loc(row, col)
                next_state, reward, done = env.step(action)
                agent.memory.add(state, action, reward, next_state, done)


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=80, fc2_units=80, env_name = ''):
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
        env_name = env_name.replace('\r', '')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.flatten(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
        self.qnetwork_local = QNetwork(state_size, action_size, seed, env_name=env_name).to(device)
        self.qnetwork_targets = deque()
        for _ in range(args.num_model):
            self.qnetwork_targets.append(QNetwork(state_size, action_size, seed, env_name=env_name).to(device))
        # create this list to maintain the order of newest to oldest target network
        self.qnetwork_targets_idx = [i for i in range(args.num_model)]
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.replace_model = 0

    def step(self):
        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
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
        """ Update model parameters.

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
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

    def take_value_estimate_v2(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            value_estimate = 0
            for k in range(args.num_model-1,-1,-1): #loop from the newest to the oldest model, in case later needed for recency-weighting avg
                value_estimate += self.qnetwork_targets[self.qnetwork_targets_idx[k]](state)
            value_estimate = value_estimate / args.num_model
            action = np.argmax(value_estimate.cpu().detach().numpy())
            value_estimate = np.max(value_estimate.cpu().detach().numpy())
        return value_estimate, action


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

def evaluate(args, agent, eps):
    start_time = time.time()
    state = env_test.reset(random_loc=False)
    score = 0; score_list = []; value_est = 0; n_steps_cur_episode = 0.0
    for i in range(args.step_size_testing): #evaluate per step_size_testing
        action = agent.act(state, eps=eps)
        next_state, reward, done = env_test.step(action)
        value_est += agent.take_value_estimate(state)
        state = next_state
        score += reward
        n_steps_cur_episode += 1.0
        if done:
            score_list.append(score)
            score = 0
            state = env_test.reset(random_loc=False)
    if len(score_list) == 0:
        mean_score = 0
    else:
        mean_score = np.mean(score_list)
        # min, max = np.min(score_list), np.max(score_list)
    end_time = time.time()
    print("\nEval. time with " + str(args.step_size_testing) + " steps: ", end_time-start_time)
    return mean_score, value_est/n_steps_cur_episode

def evaluate_new(args, agent, eps):
    start_time = time.time()
    # evaluate score and value estimate when agent only consider behav. net
    state = env_test.reset(random_loc=False)
    score = 0; score_list = []; value_est = 0; n_steps = 0.0
    for i in range(args.step_size_testing): #evaluate per step_size_testing
        action = agent.act(state, eps=eps)
        next_state, reward, done = env_test.step(action)
        value_est += agent.take_value_estimate(state)
        state = next_state
        score += reward
        n_steps += 1.0
        if done:
            score_list.append(score)
            score = 0
            state = env_test.reset(random_loc=False)
    if len(score_list) == 0:
        mean_score1 = 0
    else:
        mean_score1 = np.mean(score_list)
        # min, max = np.min(score_list), np.max(score_list)
    value_est1 = value_est / n_steps

    # evaluate score and value estimate when agent consider K net
    state = env_test.reset(random_loc=False)
    score = 0; score_list = []; value_est = 0; n_steps = 0.0
    for i in range(args.step_size_testing):  # evaluate per step_size_testing
        value_estimate, action_from_net = agent.take_value_estimate_v2(state)
        # Epsilon-greedy action selection
        if random.random() > eps:
            action = action_from_net
        else:
            action = random.choice(np.arange(env.action_space.shape[0]))
        value_est += value_estimate
        # action = agent.act(state)
        next_state, reward, done = env_test.step(action)
        state = next_state
        score += reward
        n_steps += 1.0
        if done:
            score_list.append(score)
            score = 0
            state = env_test.reset(random_loc=False)
    if len(score_list) == 0:
        mean_score2 = 0
    else:
        mean_score2 = np.mean(score_list)
    value_est2 = value_est / n_steps
    end_time = time.time()
    print("\nEval. time with " + str(args.step_size_testing) + " steps: ", end_time - start_time)
    return mean_score1, value_est1, mean_score2, value_est2

def store_json(scores1, value_ests1, scores2, value_ests2):
    # create log file
    time_now = datetime.now()
    time_string = time_now.strftime("%Y-%m-%d_%H-%M-%S")
    log = {'time': time_string, 'last_steps': LAST_FRAMES, 'last_episode': LAST_EPISODE,
           'last_score': LAST_SCORE, 'scores1': scores1, 'value_est1': value_ests1,
           'scores2': scores2, 'value_est2': value_ests2}

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

    with open('log/recent' + '_' + env_name.lower()+'_k'+str(args.num_model)+"_seed"+str(args.seed)+".json", 'w') as outfile:
        json.dump(log, outfile)


def dqn(agent, total_steps, max_t=1000, eps_start=1.0, eps_end=0.1, eps_decay=0.995): #paper use eps_end = 0.1
    """Deep Q-Learning.
    
    Params
    ======
        total_steps (float): maximum number of training frames
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    global LAST_SCORE, LAST_EPISODE, LAST_FRAMES
    scores1 = []                           # list containing scores from each episode
    scores2 = []                           # list containing scores from each episode
    scores_window = deque(maxlen=100)      # last 100 scores
    value_ests1 = []                       # list containing value estimates from each episode
    value_ests2 = []                       # list containing value estimates from each episode
    value_ests_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                        # initialize epsilon
    current_steps = 0                      # initialize total frames
    score1 = 0; value_est1 = 0
    # initialize ER buffer
    init_buffer(env, agent)
    start_time = time.time()
    from itertools import count as cnt1
    # for i_episode in range(1, n_episodes+1):
    for i_episode in cnt1():
        i_episode += 1
        state = env.reset(random_loc=False)
        n_steps_cur_episode = 0.0

        # for t in range(max_t):
        from itertools import count as cnt2
        for t in cnt2():
            action = agent.act(state, eps)
            current_eps = ((eps_end - eps_start) * (float)(current_steps) / LAST_STEP_DECREASING_EPS) + eps_start
            eps = max(eps_end, current_eps)  # use linear interpolation decay
            next_state, reward, done, = env.step(action)
            agent.step()

            if (current_steps%N_STEP_EVAL == 0): #evaluate the agent per N_STEP_EVAL
                score1, value_est1, score2, value_est2 = evaluate_new(args, agent, eps) #record two mode testing
                scores_window.append(score1)
                value_ests_window.append(value_est1)
                scores1.append(score1)  # save most recent score
                scores2.append(score2)  # save most recent score
                value_ests1.append(value_est1)  # save most recent value est.
                value_ests2.append(value_est2)  # save most recent value est.
                store_json(scores1, value_ests1, scores2, value_ests2)
                LAST_SCORE = score1
                end_time = time.time()
                print("Time per logging:", end_time-start_time, ", per", N_STEP_EVAL, "steps")
                start_time = time.time()
            state = next_state
            n_steps_cur_episode += 1.0
            current_steps += 1
            LAST_FRAMES = current_steps
            
            if current_steps % 2500 == 0:
                print('\rEnv {}\tEpisode {}\tSteps {}\tAverage Score: {:.2f}\tValue Est.: {:.2f}'.format(env_name,
                                                                                                         i_episode,
                                                                                                         current_steps,
                                                                                                         score1,
                                                                                                         value_est1))
            if done:
                break

        LAST_EPISODE = i_episode
        if (current_steps >= total_steps): #quite training after total_steps
            break

state_size = env.gridsize[0] * env.gridsize[1]
agent = Agent(state_size=state_size,
              action_size=env.action_space.shape[0],
              seed=args.seed,
              env_name=env_name)
dqn(agent, total_steps=args.total_steps)