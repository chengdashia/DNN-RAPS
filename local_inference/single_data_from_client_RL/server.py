from communication.communicator import NodeEnd
from utils import get_client_app_port_by_name


def start():
    # 建立连接
    node = NodeEnd(host_ip, host_port)

    # 设置循环次数
    num_iterations = 5

    for i in range(num_iterations):
        print(f"Iteration {i + 1}")
        global info

        # 初始化分割点
        node_layer_indices = {'client2': [0, 1], 'client1': [2, 3], 'client3': [4, 5, 6]}

        # 向第一个客户端发送分割点信息
        first_client = list(node_layer_indices.keys())[0]
        first_ip, first_port = get_client_app_port_by_name(first_client, model_name)

        # 连接第一个客户端
        node.__init__(host_ip, host_port)  # 重新初始化服务端连接
        node.node_connect(first_ip, first_port)

        # 发送分割点信息
        msg = ["开始推理", node_layer_indices]
        node.send_message(node.sock, msg)
        print(f"服务端将分割点信息发送到客户端{first_ip}:{first_port}")
        node.sock.close()

        # 等待所有客户端执行完毕
        execution_times = {}
        for client in node_layer_indices.keys():
            node.__init__(host_ip, host_port)  # 重新初始化服务端连接
            connect_socket, _ = node.wait_for_connection()
            msg = node.receive_message(connect_socket)
            execution_times[client] = msg[0]
            node.sock.close()

        # 输出每个客户端的执行时间
        for client, time in execution_times.items():
            print(f"客户端{client}执行时间: {time:.4f}s")
        print()


if __name__ == '__main__':

    host_ip = '127.0.0.1'
    host_port = 9000

    model_name = 'VGG5'

    info = "MSG_FROM_NODE_ADDRESS(%s), port= %s" % (host_ip, host_port)

    start()
