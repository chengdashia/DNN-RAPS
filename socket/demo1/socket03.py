import socket
import json

# 假设客户端知道自己的IP和端口
CLIENT_IP = 'localhost'
CLIENT_PORT = 9003  # 这个端口需要对每个客户端实例进行更改

def process_data(data):
    # 在这里实现数据处理逻辑
    return data.upper()

def get_client_index(clients, client_info):
    # 使用 next() 和 enumerate() 获取满足条件的第一个匹配的索引
    index = next(i for i, server in enumerate(clients) if tuple(server) == client_info)
    return index

def client_operation(client_info):
    # 连接到服务器
    server_address = ('localhost', 9002)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)

    try:
        # 接收数据
        received_data = sock.recv(4096)
        received_json = json.loads(received_data.decode('utf-8'))

        # 获取客户端列表和数据
        clients_list = received_json["clients"]
        data = received_json["data"]

        print(clients_list)
        print(client_info)

        # 确定自己在客户端列表中的位置
        client_index = get_client_index(clients_list, client_info)

        # 处理数据
        processed_data = process_data(data)
        received_json["data"] = processed_data

        # 确定下一个客户端的地址
        next_index = (client_index + 1) % len(clients_list)
        next_client_info = tuple(clients_list[next_index])

        print("next_client_info  ", next_client_info)

        # 发送数据到下一个客户端或回服务器
        if client_info == clients_list[-1]:  # 如果是最后一个客户端，则发送回服务器
            sock.sendall(json.dumps(received_json).encode('utf-8'))
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as next_sock:
                next_sock.connect(next_client_info)
                next_sock.sendall(json.dumps(received_json).encode('utf-8'))
                print(f"Data sent to the next client: {next_client_info}")

    finally:
        sock.close()

if __name__ == "__main__":
    client_info = (CLIENT_IP, CLIENT_PORT)
    client_operation(client_info)
