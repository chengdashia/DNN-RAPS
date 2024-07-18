import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from resource_utilization import get_linux_resource_info


# 模拟的VGG5模型推理时间
def client_inference_time(split_point):
    return split_point * 0.1


def server_inference_time(split_point):
    return (7 - split_point) * 0.1


def transmission_time(split_point, bandwidth):
    return (split_point * 1000) / bandwidth  # 假设每层的数据大小为1000字节


class EdgeEnv:
    def __init__(self, server_info):
        self.state = [
            server_info['cpu'],
            server_info['memory'],
            server_info['network']
        ]
        self.action_space = [i for i in range(1, 8)]  # 划分点从1到7
        self.state_space = len(self.state)
        self.reset()

    def reset(self):
        return self.state

    def step(self, action):
        client_time = client_inference_time(action)
        trans_time = transmission_time(action, self.state[2])
        server_time = server_inference_time(action)

        total_time = client_time + trans_time + server_time
        reward = -total_time

        return self.state, reward, False, {}


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)
        state = torch.FloatTensor(state).to(self.device)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            action = torch.LongTensor([action]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            current_q = self.model(state)[action]
            loss = self.criterion(current_q, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


if __name__ == "__main__":
    server_ip = "your_server_ip"  # 替换为实际的服务器 IP
    server_info = get_linux_resource_info(server_ip)
    if not server_info:
        print("Failed to retrieve server information.")
        exit()

    env = EdgeEnv(server_info)
    state_size = env.state_space
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            agent.replay(batch_size)

        if e % 10 == 0:
            print(f"episode: {e}/{episodes}, score: {reward}, epsilon: {agent.epsilon:.2}")

    print("Training finished.")
    agent.save("dqn_vgg5_split_model.pth")

