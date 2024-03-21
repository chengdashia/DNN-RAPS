from communicator import NodeEnd


def start(host_ip, host_port, next_ip, next_port):
    # 建立连接
    node = NodeEnd(host_ip, host_port)

    # 连接服务端
    node.node_connect(server_ip, server_port)

    # 接收服务端发送的数据
    msg = node.receive_message(node.sock)
    print(f"客户端{host_ip}:{host_port}接收到消息: {msg}")

    # 关闭与服务端的连接
    node.sock.close()

    # 对数据进行处理
    processed_msg = process_data(msg)

    # 创建一个新的连接来连接下一个客户端
    next_node = NodeEnd(host_ip, host_port)
    next_node.node_connect(next_ip, next_port)

    # 将处理后的数据发送给下一个客户端
    next_node.send_message(next_node.sock, processed_msg)
    print(f"客户端{host_ip}:{host_port}将处理后的消息发送到客户端{next_ip}:{next_port}")

    # 关闭与下一个客户端的连接
    next_node.sock.close()


def process_data(msg):
    # 在这里对数据进行处理,这里只是简单地在消息前面加上"processed_by_client1_"
    processed_msg = ["processed_by_client1_" + str(m) for m in msg]
    return processed_msg


if __name__ == '__main__':
    server_ip = '127.0.0.1'
    server_port = 9000

    host_ip = '127.0.0.1'
    host_port = 9001
    next_ip = '127.0.0.1'
    next_port = 9002

    start(host_ip, host_port, next_ip, next_port)