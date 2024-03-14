import socket
import json
import threading

# 初始化服务器信息
server_address = ('localhost', 9001)
clients_info = [
    ('localhost', 9002),
    ('localhost', 9003)
]

# 创建 JSON 数据
data_to_send = {
    "clients": [server_address] + clients_info,
    "data": "msg"
}

def handle_client(connection, client_address):
    try:
        print(f"Connection from {client_address}")

        # 发送数据到客户端
        message = json.dumps(data_to_send).encode('utf-8')
        connection.sendall(message)

        # 等待来自最后一个客户端的响应
        final_response = connection.recv(4096)
        print(f"Received response from the client: {final_response.decode('utf-8')}")
    finally:
        connection.close()

def send_to_clients():
    # 创建 TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定 socket 到地址
    sock.bind(server_address)
    # 监听连接
    sock.listen()

    print(f"Listening for connections on {server_address[0]}:{server_address[1]}...")

    while True:
        # 等待连接
        connection, client_address = sock.accept()
        # 创建新线程来处理客户端
        client_thread = threading.Thread(target=handle_client, args=(connection, client_address))
        client_thread.start()

if __name__ == "__main__":
    send_to_clients()