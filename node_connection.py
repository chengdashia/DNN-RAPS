import socket
import time
from communicator import Communicator


class NodeConnection(Communicator):
    def __init__(self, ip, port):
        # 调用父类communicator的构造函数来初始化
        super(NodeConnection, self).__init__(ip, port)

    def add_addr(self, node_addr, node_port, max_retries=10):
        # 尝试连接的次数
        attempts = 0
        while attempts < max_retries:
            try:
                # 尝试创建到下一个节点的连接
                self.sock.connect((node_addr, node_port))
                # 如果连接成功，则跳出循环
                print(f"已成功连接到{node_addr}:{node_port}")
                # If the connection is successful, break the loop
                break
            except socket.error as e:
                # 如果连接失败，打印错误消息并重试
                print(f"连接到{node_addr}:{node_port}失败，正在重试...错误：{e}")
                # 等待一段时间再次尝试
                time.sleep(1)
                # 增加尝试连接的次数
                attempts += 1
        if attempts == max_retries:
            # 如果达到最大重试次数，则抛出异常
            raise Exception(f"无法连接到{node_addr}:{node_port}， 已达到最大重试次数{max_retries}")