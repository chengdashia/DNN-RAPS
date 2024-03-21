from communicator import NodeEnd


def start():
    # 建立连接
    node = NodeEnd(host_ip, host_port)
    # 准备发送的消息内容
    msg = [info, node_layer_indices, layer_node_indices]
    # 等待客户端连接
    connect_socket, _ = node.wait_for_connection()
    # 发送信息给客户端
    node.send_message(connect_socket, msg)
    print(f"服务端{host_ip}:{host_port}将消息发送到客户端")
    # 关闭连接
    node.sock.close()


if __name__ == '__main__':
    host_ip = '127.0.0.1'
    host_port = 9000
    model_name = 'VGG5'
    node_layer_indices = {'127.0.0.1:9001': [0, 1], '127.0.0.1:9002': [2, 3], '127.0.0.1:9003': [4, 5, 6]}
    layer_node_indices = {
              0: '127.0.0.1',
              1: '127.0.0.1',
              2: '127.0.0.1',
              3: '127.0.0.1',
              4: '127.0.0.1',
              5: '127.0.0.1',
              6: '127.0.0.1'
      }

    info = "MSG_FROM_NODE_ADDRESS(%s), host= %s" % (host_ip, host_port)
    start()
