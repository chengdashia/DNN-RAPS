import socket
import threading
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from network_utils import send_data, receive_data
from config import B, dataset_path, iterations


def get_next_client(current_client):
    """
    获取下一个客户端
    :param current_client:  当前客户端
    :return:              下一个客户端
    """
    keys = list(node_layer_indices.keys())
    try:
        current_index = keys.index(current_client)
        next_index = current_index + 1
        if next_index < len(keys):
            return keys[next_index]
        return None
    except ValueError:
        return None


def prepare_data():
    """
    加载数据集并返回数据和标签
    :return:  数据集和标签
    """
    data_dir = dataset_path
    test_dataset = datasets.CIFAR10(
        data_dir,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        download=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=B, shuffle=False, num_workers=0
    )
    data_cpu_list, target_cpu_list = [], []
    for data, target in test_loader:
        data_cpu_list.append(data)
        target_cpu_list.append(target)
    return data_cpu_list, target_cpu_list


def calculate_accuracy(fx, y):
    """
    计算模型输出与真实标签之间的准确率

    参数:
    fx (Tensor): 模型输出
    y (Tensor): 真实标签

    返回值:
    acc (float): 准确率(0-100)
    """
    # 计算预测值，fx是模型输出，y是真实标签
    predictions = fx.max(1, keepdim=True)[1]
    # 将预测值和真实标签转换为相同形状
    correct = predictions.eq(y.view_as(predictions)).sum()
    # 计算准确率，correct是预测正确的样本数量
    acc = 100.00 * correct.float() / predictions.shape[0]
    return acc


def get_loss_acc(result_list):
    """
    计算结果的loss和acc
    :param result_list:    客户端的推理结果
    :return:              loss和acc
    """
    for i in range(iterations):
        loss = F.cross_entropy(result_list[i], target_list[i])
        acc = calculate_accuracy(result_list[i], target_list[i])
        loss_list.append(loss)
        acc_list.append(acc)
    print("loss :{:.4}".format(sum(loss_list) / len(loss_list)))
    print("acc :{:.4}%".format(sum(acc_list) / len(acc_list)))
    print("")


def handle_client(conn, client_name):
    """
    处理客户端的请求
    :param conn:          客户端的连接
    :param client_name:  客户端的名称
    :return:              None
    """
    for iteration in range(1, max_iterations + 1):
        print(f"======================  Start processing round {iteration} ================================")
        message = receive_data(conn)
        if not message:
            continue

        inference_time, data_inference_list, processed_cumulative_layer_number = message
        print(f"Processing time from {client_name}: {inference_time} seconds")
        # print(f"processed_data_cpu_list from {client_name}: {processed_data_cpu_list}")
        # print(f"processed_target_cpu_list from {client_name}: {processed_target_cpu_list}")

        next_client_name = get_next_client(client_name)
        # 如果是最后一个客户端 则计算acc和loss。同时数据初始化
        if next_client_name is None:
            # 计算结果
            get_loss_acc(data_inference_list)

            # 重新初始化
            print("Round completed. Restarting with initial data.")
            data_inference_list = data_list
            processed_cumulative_layer_number = cumulative_layer_number
            next_client_name = list(node_layer_indices.keys())[0]

        next_client_conn = clients[next_client_name]
        # 序列化数据后发送
        data_to_send = [node_layer_indices, data_inference_list, processed_cumulative_layer_number]
        send_data(next_client_conn, data_to_send)


def accept_connections(server_socket):
    """
    接受并处理客户端的连接
    :param server_socket:  服务器套接字
    :return:              None
    """
    # 等待所有的客户端连接
    while len(clients) < len(node_layer_indices):
        conn, addr = server_socket.accept()
        client_name = receive_data(conn)
        clients[client_name] = conn
        print(f"{client_name} connected from {addr}")
        threading.Thread(target=handle_client, args=(conn, client_name)).start()

    first_client_name = list(node_layer_indices.keys())[0]
    # 消息封装
    data_to_send = [node_layer_indices, data_list, cumulative_layer_number]
    # 序列化数据后发送
    send_data(clients[first_client_name], data_to_send)


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建一个 TCP 套接字，使用 IPv4 地址
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 设置套接字选项，允许重新使用地址
    server_socket.bind(('localhost', 9000))  # 将套接字绑定到 localhost 上的 12345 端口
    server_socket.listen()  # 配置套接字进入监听状态，等待客户端连接
    print("Server is listening for connections...")
    accept_connections(server_socket)  # 调用 accept_connections 函数来接受并处理连接


if __name__ == "__main__":
    # 定义初始数据
    node_layer_indices = {'client1': [0, 1], 'client2': [2, 3], 'client3': [4, 5, 6]}

    model_name = 'VGG5'
    # 从数据集加载数据
    data_list, target_list = prepare_data()
    # 存储连接的客户端
    clients = {}
    # 设定最大迭代次数
    max_iterations = 10

    cumulative_layer_number = 0

    # 损失列表
    loss_list = []
    # 准确率列表
    acc_list = []

    main()
