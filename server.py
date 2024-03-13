"""
    服务器执行文件，主要负责：
        1、根据客户端的资源使用情况。将模型文件进行分层
        ip及对应节点位序
"""
import torch
from torch import nn
import logging
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config
from node_end import NodeEnd
from models.vgg.vgg import VGG
from models.model_struct import model_cfg
from utils.segmentation_strategy import NetworkSegmentationStrategy
from utils.resource_utilization import get_all_server_info


def segment_network(name, split_points):
    """
    根据选取的分割点,将模型划分为多个segment
    :param name: 模型名称
    :param split_points: 分割点的列表
    :return:
    """
    segments = []
    start = 0
    for point in split_points:
        segments.append(model_cfg[name][start:point])
        start = point
    # Add the last segment
    segments.append(model_cfg[name][start:])
    # Print the layer configurations for each segment and the segmentation points
    logging.info("Model segments:")
    for i, segment in enumerate(segments):
        logging.info(f"Segment {i + 1}:")
        for layer_cfg in segment:
            logging.info(layer_cfg)
        # New line for better readability between segments
        logging.info("\n")
    return segments


def get_layer_indices(split_models):
    """
    在每个节点上计算的第k层
    :param split_models:
    :return: split_layers, reverse_split_layers
    """
    # 键为节点索引,值为该节点上计算的层索引列表
    split_layers = {}
    # 键为层索引,值为该层所在节点的索引
    reverse_split_layers = {}
    start_index = 0
    for i, model in enumerate(split_models):
        layer_indices = list(range(start_index, start_index + len(model)))
        split_layers[i] = layer_indices
        for j in layer_indices:
            reverse_split_layers[j] = i
        start_index += len(model)
    # split_layer = {0: [0, 1], 1: [2, 3], 2: [4, 5, 6]}
    # reverse_split_layer = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
    logging.info("split_layers: %s", split_layers)
    logging.info("reverse_split_layers: %s", reverse_split_layers)
    return split_layers, reverse_split_layers


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


def node_inference(node, model):
    """
    节点推理的主要逻辑。它接收来自其他节点的数据和信息，计算输出，然后将结果发送给下一个节点。如果是最后一层，它会计算损失。
    :param node:
    :param model:
    :return:
    """
    # 重新初始化节点
    node.__init__(host_ip, host_port)
    while True:
        global reverse_split_layer, split_layer
        # 存储已发送的IP地址
        last_send_ips = []
        # 迭代次数
        iteration = int(config.N / config.B)
        # 等待连接
        node_socket, node_addr = node.wait_for_connection()
        # 迭代处理每一批数据
        for i in range(iteration):
            logging.info(f"node_{host_node_num} 获取来自 node{node_addr} 的连接")
            msg = node.receive_message(node_socket)
            logging.info("msg: %s", msg)
            # 解包消息内容
            data, target, start_layer, split_layer, reverse_split_layer = msg
            # 计算输出
            data, next_layer, split = calculate_output(model, data, start_layer)
            # 如果不是最后一层
            if split + 1 < model_len:
                # 获取下一个节点的IP
                last_send_ip = config.CLIENTS_LIST[reverse_split_layer[split + 1]]
                if last_send_ip not in last_send_ips:
                    # 添加到发送列表
                    node.connect(last_send_ip, 1998)
                last_send_ips.append(last_send_ip)
                msg = [info, data.cpu(), target.cpu(), next_layer, split_layer, reverse_split_layer]
                node.send_message(node.sock, msg)
                print(
                    f"node_{host_node_num} send msg to node{config.CLIENTS_LIST[reverse_split_layer[split + 1]]}"
                )
            else:
                # 到达最后一层，计算损失
                loss = torch.nn.functional.cross_entropy(data, target)
                loss_list.append(loss)
                print("loss :{}".format(sum(loss_list) / len(loss_list)))
                print("")

        # 关闭socket连接
        node_socket.close()
        # 重新初始化节点
        node.__init__(host_ip, host_port)


def get_model(model, type, in_channels, out_channels, kernel_size, start_layer):
    """
     获取当前节点需要计算的模型层

     参数:
     model (nn.Module): 模型
     type (str): 层类型('M':池化层, 'D':全连接层, 'C':卷积层)
     in_channels (int): 输入通道数
     out_channels (int): 输出通道数
     kernel_size (int): 卷积核大小
     start_layer (int): 起始层索引

     返回值:
     features (nn.Sequential): 卷积层和池化层
     dense_s (nn.Sequential): 全连接层
     next_layer (int): 下一层索引
     """
    features = []
    dense_s = []
    if type == "M":
        features.append(model.features[start_layer])
        start_layer += 1
    if type == "D":
        # modify the start_layer
        dense_s.append(model.denses[start_layer-11])
        start_layer += 1
    if type == "C":
        for i in range(3):
            features.append(model.features[start_layer])
            start_layer += 1
    next_layer = start_layer
    return nn.Sequential(*features), nn.Sequential(*dense_s), next_layer


def calculate_output(model, data, start_layer):
    """
    计算当前节点的输出

    参数:
    model (nn.Module): 模型
    data (Tensor): 输入数据
    start_layer (int): 起始层索引

    返回值:
    data (Tensor): 输出数据
    next_layer (int): 下一层索引
    split (int): 当前节点计算的最后一层索引
    """
    output = data
    split = None
    # 初始化next_layer为start_layer
    next_layer = start_layer
    for i, layer_idx in enumerate(split_layer[host_node_num]):
        # TODO:如果节点上的层不相邻，需要兼容
        layer_type = model_cfg[model_name][layer_idx][0]
        in_channels = model_cfg[model_name][layer_idx][1]
        out_channels = model_cfg[model_name][layer_idx][2]
        kernel_size = model_cfg[model_name][layer_idx][3]
        # print("type,in_channels,out_channels,kernel_size",type,in_channels,out_channels,kernel_size)
        features, dense, next_layer = get_model(
            model, layer_type, in_channels, out_channels, kernel_size, start_layer
        )
        if len(features) > 0:
            model_layer = features
        else:
            model_layer = dense
        # 计算输出
        output = model_layer(output)
        # 更新当前节点计算的最后一层索引
        split = layer_idx
    return data, next_layer, split


def start_inference():
    """
    整个推理过程的入口。
    它初始化模型和节点连接，如果包含第一层，它会加载数据集并计算第一层的输出，然后将结果发送给下一个节点。
    最后，它调用 node_inference 函数开始节点推理过程。
    """
    include_first = True
    # 建立连接
    node = NodeEnd(host_ip, host_port)
    model = VGG("Client", model_name, 6, model_cfg)
    model.eval()
    model.load_state_dict(torch.load("models/vgg/vgg.pth"))

    # 如果含第一层，载入数据
    if include_first:
        # TODO:modify the data_dir
        start_layer = 0
        data_dir = "dataset"
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
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )

        last_send_ips = []
        for data, target in test_loader:
            # print(len(data))
            # split:当前节点计算的层
            # next_layer:下一个权重层
            data, next_layer, split = calculate_output(model, data, start_layer)

            # TODO:modify the port
            last_send_ip = config.CLIENTS_LIST[reverse_split_layer[split + 1]]
            if last_send_ip not in last_send_ips:
                node.connect(last_send_ip, 1998)
            last_send_ips.append(last_send_ip)

            # TODO:是否发送labels
            msg = [info, data.cpu(), target.cpu(), next_layer, split_layer, reverse_split_layer]
            print(
                f"node{host_node_num} send msg to node{config.CLIENTS_LIST[reverse_split_layer[split + 1]]}"
            )
            node.send_message(node.sock, msg)
            include_first = False
            # print('*' * 40)
        node.sock.close()
    node_inference(node, model)


if __name__ == '__main__':

    model_name = "VGG5"
    model_len = len(model_cfg[model_name])

    # 获取所有节点的资源情况
    nodes_resource_infos = get_all_server_info(config.server_list)

    # 根据不同的分割策略,选取分割点
    segmentation_strategy = NetworkSegmentationStrategy(model_name, model_cfg, nodes_resource_infos)
    segmentation_points = segmentation_strategy.random_select_segmentation_points()
    print('*' * 40)
    print("segmentation_points: ", segmentation_points)

    # 根据选取的分割点分割网络
    segmented_models = segment_network(model_name, segmentation_points)
    print("segmented_models", segmented_models)

    split_layer, reverse_split_layer = get_layer_indices(segmented_models)

    host_port = 9001
    host_node_num = 0
    host_ip = config.CLIENTS_LIST[host_node_num]

    info = "MSG_FROM_NODE(%d), host= %s" % (host_node_num, host_ip)

    loss_list = []

    # 开始推理
    start_inference()
