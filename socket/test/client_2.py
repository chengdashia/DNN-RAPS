from communicator import NodeEnd


def start(host_ip, host_port):
    # 建立连接
    node = NodeEnd(host_ip, host_port)

    # 等待上一个客户端的连接
    connect_socket, _ = node.wait_for_connection()

    # 接收上一个客户端发送的数据
    msg = node.receive_message(connect_socket)
    print(f"客户端{host_ip}:{host_port}接收到消息: {msg}")

    # 对数据进行处理
    processed_msg = process_data(msg)

    print(f"客户端{host_ip}:{host_port}处理后的消息: {processed_msg}")

    # 关闭连接
    node.sock.close()


def process_data(msg):
    # 在这里对数据进行处理,这里只是简单地在消息前面加上"processed_by_client2_"
    processed_msg = ["processed_by_client2_" + str(m) for m in msg]
    return processed_msg


if __name__ == '__main__':
    host_ip = '127.0.0.1'
    host_port = 9002

    start(host_ip, host_port)