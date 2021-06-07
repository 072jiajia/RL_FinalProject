
import argparse
import sys
import copy
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser(description='CVAE')
parser.add_argument('--env', type=str, default="CartPole-v0", help='Open AI environment')
parser.add_argument('--Episode', default=400, type=int, help='number of episode to learn')
parser.add_argument('--K', type=int, default=5, help='average size')
args = parser.parse_args()

Variable = torch.FloatTensor

class Network(nn.Module):
    def __init__(self, n_in, n_out):
        super(Network, self).__init__()
        self.L1 = nn.Linear(n_in, 100)
        self.L2 = nn.Linear(100, 200)
        self.L3 = nn.Linear(200, 100)
        self.L4 = nn.Linear(100, 100)
        self.q_value = nn.Linear(100, n_out)
        self.q_value.weight.data.fill_(0.)

    def q_func(self, in_layer):
        layer1 = F.leaky_relu(self.L1(in_layer))
        layer2 = F.leaky_relu(self.L2(layer1))
        layer3 = F.leaky_relu(self.L3(layer2))
        layer4 = F.leaky_relu(self.L4(layer3))
        return self.q_value(layer4)


class Agent():
    def __init__(self, n_state, n_action, seed):
        np.random.seed(seed)
        self.n_action = n_action

        self.model = Network(n_state, n_action)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.memory = deque()
        self.step = 0
        self.train_freq = 10
        self.target_update_freq = 20
        self.gamma = 0.99
        self.mem_size = 1000
        self.replay_size = 100
        self.epsilon = 0.05

    def stock_experience(self, exp):
        self.memory.append(exp)
        if len(self.memory) > self.mem_size:
            self.memory.popleft()

    def forward(self, exp, target_model):
        state = Variable(exp["state"])
        state_dash = Variable(exp["state_dash"])

        # forward and reshape it because I don't know why 
        # in this part 
        # In chainer, (100, 1, 4) get output (100, 2)  <<-- squeezed
        # But in torch, (100, 1, 4) get output (100, 1, 2)
        # so I do q_action = q_action[:, 0, :]
        q_action = self.model.q_func(state)
        q_action = q_action[:, 0, :]

        tmp = 0
        for i in range(args.K):
            tmp += target_model[i].q_func(state_dash)
        tmp = tmp / args.K
        tmp = tmp[:, 0, :]

        max_q_dash = torch.max(tmp.data, dim=-1)[0]

        target = copy.deepcopy(q_action.data)
        for i in range(self.replay_size):
            target[i, exp["action"][i]] = exp["reward"][i] + (self.gamma * max_q_dash[i]) * (not exp["ep_end"][i])

        loss = F.mse_loss(q_action, Variable(target))
        return loss

    def action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_action)
        else:
            state = Variable(state)
            q_action = self.model.q_func(state)
            q_action = q_action.data[0]
            act = np.argmax(q_action)
            return np.asarray(act, dtype=np.int8)

    def experience_replay(self, target_model):
        mem = np.random.permutation(np.array(self.memory))
        perm = np.array([i for i in range(len(mem))])
        tmp_loss = 0
        for start in perm[::self.replay_size]:
            index = perm[start: start+self.replay_size]
            replay = mem[index]

            # generate batch of experience
            state = np.array([replay[i]["state"] for i in range(self.replay_size)], dtype=np.float32)
            action = np.array([replay[i]["action"] for i in range(self.replay_size)], dtype=np.int8)
            reward = np.array([replay[i]["reward"] for i in range(self.replay_size)], dtype=np.float32)
            state_dash = np.array([replay[i]["state_dash"] for i in range(self.replay_size)], dtype=np.float32)
            ep_end = np.array([replay[i]["ep_end"] for i in range(self.replay_size)], dtype=np.bool)
            experience = {"state": state, "action": action, "reward": reward, "state_dash": state_dash, "ep_end": ep_end}

            self.model.zero_grad()
            loss = self.forward(experience, target_model)
            tmp_loss += loss.data
            loss.backward()
            self.optimizer.step()
        return tmp_loss / self.replay_size

    def train(self, target_model):
        loss = -1
        if len(self.memory) >= self.mem_size:
            if self.step % self.train_freq == 0:
                loss = self.experience_replay(target_model)
                target_model.popleft()
                target_model.append(copy.deepcopy(agent.model))
        self.step += 1
        return target_model, loss

env = gym.make(args.env)
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
seed = 123
agent = Agent(n_state, n_action, seed)
target_model = deque()
for i in range(args.K):
    target_model.append(copy.deepcopy(agent.model))

for _episode in range(args.Episode):

    observation = env.reset()
    total_reward = 0
    loss = 0
    count = 0
    for _times in range(200):  # 2000
        # env.render()
        [target_model, tmp_loss] = agent.train(target_model)
        if tmp_loss >= 0:
            loss += tmp_loss
            count += 1
        state = observation.astype(np.float32).reshape((1, n_state))
        action = agent.action(state)
        observation, reward, ep_end, _ = env.step(action)
        state_dash = observation.astype(np.float32).reshape((1, n_state))
        experience = {"state": state, "action": action, "reward": reward, "state_dash": state_dash, "ep_end": ep_end}
        agent.stock_experience(experience)
        total_reward += reward
        if ep_end:
            if count == 0:
                ans = None
            else:
                ans = loss / count
            print(_episode, ans, total_reward)
            break
