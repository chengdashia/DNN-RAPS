"""
    服务器执行文件，主要负责：
        1、根据客户端的资源使用情况。将模型文件进行分层
        ip及对应节点位序
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config
from node_connection import NodeConnection
from models.vgg.vgg import VGG
from models.model_struct import model_cfg
from utils.segmentation_strategy import NetworkSegmentationStrategy


# 根据选取的分割点 分割网络
def segment_network(split_points):
    segments = []
    start = 0
    for point in split_points:
        segments.append(model_cfg['VGG5'][start:point])
        start = point
    segments.append(model_cfg['VGG5'][start:])  # Add the last segment
    # Print the layer configurations for each segment and the segmentation points
    for i, segment in enumerate(segments):
        print(f"Segment {i + 1}:")
        for layer_cfg in segment:
            print(layer_cfg)
        print("\n")  # New line for better readability between segments
    return segments


# 在每个节点上计算的第k层
def segmented_index(split_models):
    split_layers = {}
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
    print("split_layers: ", split_layers)
    print("reverse_split_layers: ", reverse_split_layers)
    return split_layers, reverse_split_layers


# 计算准确度
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    # print("preds={}, y.view_as(preds)={}".format(preds, y.view_as(preds)))
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


# 节点推理
def node_inference(node, model):
    node.__init__(host_ip, host_port)
    while True:
        global reverse_split_layer, split_layer
        last_send_ips = []
        iteration = int(config.N / config.B)
        node_socket, node_addr = node.wait_for_connection()
        for i in range(iteration):
            print("node_{host_node_num} get connection from node{node_addr}")
            msg = node.receive_message(node_socket)
            print("msg:  ", msg)
            data = msg[1]
            target = msg[2]
            start_layer = msg[3]
            split_layer = msg[4]
            reverse_split_layer = msg[5]
            data, next_layer, split = calculate_output(model, data, start_layer)
            if split + 1 < model_len:
                last_send_ip = config.CLIENTS_LIST[reverse_split_layer[split + 1]]
                if last_send_ip not in last_send_ips:
                    node.add_addr(last_send_ip, 1998)
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
        node_socket.close()
        node.__init__(host_ip, host_port)


# 获取模型
def get_model(model, type, in_channels, out_channels, kernel_size, start_layer):
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


# 计算输出
def calculate_output(model, data, start_layer):
    for split in split_layer[host_node_num]:
        # TODO:如果节点上的层不相邻，需要兼容
        type = model_cfg[model_name][split][0]
        in_channels = model_cfg[model_name][split][1]
        out_channels = model_cfg[model_name][split][2]
        kernel_size = model_cfg[model_name][split][3]
        # print("type,in_channels,out_channels,kernel_size",type,in_channels,out_channels,kernel_size)
        features, dense, next_layer = get_model(
            model, type, in_channels, out_channels, kernel_size, start_layer
        )
        if len(features) > 0:
            model_layer = features
        else:
            model_layer = dense

        data = model_layer(data)
        start_layer = next_layer
        # print("next_layer", next_layer)
    return data, next_layer, split


# 开始推理
def start_inference():
    include_first = True
    node = NodeConnection(host_ip, host_port)
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
            last_send_ip=config.CLIENTS_LIST[reverse_split_layer[split + 1]]
            if last_send_ip not in last_send_ips:
                node.add_addr(last_send_ip, 1998)

            last_send_ips.append(last_send_ip)

            # TODO:是否发送labels
            msg = [info, data.cpu(), target.cpu(), next_layer,split_layer,reverse_split_layer]
            print(
                f"node{host_node_num} send msg to node{config.CLIENTS_LIST[reverse_split_layer[split + 1]]}"
            )
            node.send_message(node.sock, msg)
            include_first = False
            # print('*' * 40)
        node.sock.close()
    node_inference(node, model)


if __name__ == '__main__':
    # 选取分割点
    segmentation_strategy = NetworkSegmentationStrategy(model_cfg)
    segmentation_points = segmentation_strategy.random_select_segmentation_points()
    print('*' * 40)
    print("segmentation_points: ", segmentation_points)

    # 根据选取的分割点分割网络
    segmented_models = segment_network(segmentation_points)
    print("segmented_models", segmented_models)

    split_layer, reverse_split_layer = segmented_index(segmented_models)

    host_port = 9001
    host_node_num = 0
    host_ip = config.CLIENTS_LIST[host_node_num]

    info = "MSG_FROM_NODE(%d), host= %s" % (host_node_num, host_ip)

    loss_list = []

    model_name = "VGG5"
    model_len = len(model_cfg[model_name])

    # 开始推理
    start_inference()