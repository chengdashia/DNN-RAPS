import numpy as np
import time


class EdgeComputingEnv:
    def __init__(self, model, model_cfg, client_resources, server_resources):
        self.model = model
        self.model_cfg = model_cfg
        self.client_resources = client_resources
        self.server_resources = server_resources
        self.state = None
        self.done = False

    def reset(self):
        self.state = np.array(self.client_resources)
        self.done = False
        return self.state

    def step(self, action):
        # action 是分割点
        split_point = action
        client_layers = self.model_cfg[:split_point]
        server_layers = self.model_cfg[split_point:]

        # 模拟客户端执行时间
        client_exec_time = self.simulate_client_execution(client_layers)

        # 模拟传输时间
        transfer_time = self.simulate_transfer_time()

        # 模拟服务端执行时间
        server_exec_time = self.simulate_server_execution(server_layers)

        total_time = client_exec_time + transfer_time + server_exec_time

        # 定义奖励为负的总时间，以最小化总推理时间
        reward = -total_time
        self.done = True
        return self.state, reward, self.done, {}

    def simulate_client_execution(self, client_layers):
        # 模拟客户端执行时间
        start_time = time.time()
        # 模拟客户端层的推理（这里只是时间的模拟）
        exec_time = np.random.rand()  # 使用实际的执行时间或模拟的时间
        end_time = time.time()
        return end_time - start_time + exec_time

    def simulate_transfer_time(self):
        # 模拟数据传输时间
        transfer_time = np.random.rand()  # 使用实际的带宽或模拟的时间
        return transfer_time

    def simulate_server_execution(self, server_layers):
        # 模拟服务端执行时间
        start_time = time.time()
        # 模拟服务端层的推理（这里只是时间的模拟）
        exec_time = np.random.rand()  # 使用实际的执行时间或模拟的时间
        end_time = time.time()
        return end_time - start_time + exec_time
