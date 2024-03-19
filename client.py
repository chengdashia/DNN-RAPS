import torch
from torch import nn
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config
from node_end import NodeEnd
from models.vgg.vgg import VGG
from models.model_struct import model_cfg
from utils.utils import get_client_app_port


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


def get_model(model, layer_type, in_channels, out_channels, kernel_size, start_layer):
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
     feature_seq (nn.Sequential): 卷积层和池化层
     dense_s (nn.Sequential): 全连接层
     next_layer (int): 下一层索引
     """
    # 存储特征层（卷积层或池化层）的序列
    feature_seq = []
    # 存储全连接层的序列
    dense_seq = []
    # 根据层的类型添加对应的层到序列中
    if layer_type == "M":
        # 如果是池化层，增加到特征层序列中
        feature_seq.append(model.features[start_layer])
        start_layer += 1
    elif layer_type == "D":
        # 如果是全连接层，增加到全连接层序列中
        dense_seq.append(model.denses[start_layer - 11])
        start_layer += 1
    elif layer_type == "C":
        # 如果是卷积层，增加连续三个卷积层到特征层序列中
        for _ in range(3):
            feature_seq.append(model.features[start_layer])
            start_layer += 1
    # 更新下一层的起始索引
    next_layer = start_layer
    # 创建特征层和全连接层的 Sequential 容器
    return nn.Sequential(*feature_seq), nn.Sequential(*dense_seq), next_layer


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

    # 定义变量，并给予默认值
    next_layer = start_layer
    split = None

    # 遍历当前主机节点上的层
    for split in node_layer_indices[host_ip]:
        # 如果节点上的层不相邻，需要实现层之间的兼容性
        layer_type = model_cfg[model_name][split][0]  # 层的类型
        in_channels = model_cfg[model_name][split][1]  # 输入通道数
        out_channels = model_cfg[model_name][split][2]  # 输出通道数
        kernel_size = model_cfg[model_name][split][3]  # 卷积核大小

        # 获取模型的当前层
        features, dense, next_layer = get_model(
            model, layer_type, in_channels, out_channels, kernel_size, start_layer
        )

        # 选择特征层还是全连接层
        model_layer = features if len(features) > 0 else dense

        # 如果是全连接层，需要先将数据展平
        if layer_type == "D":
            data = data.view(data.size(0), -1)

        # 将数据通过模型层进行前向传播
        data = model_layer(data)

        # 更新下一层的起始层索引
        start_layer = next_layer

    # 在循环结束后，确保split有一个合理的值
    if split is None:
        # 如果循环没有执行，设置split为合理的默认值
        split = start_layer  # 或者根据需要设置其他默认值

    return data, next_layer, split


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
        # 迭代次数 N为数据总数，B为批次大小
        iterations = int(config.N / config.B)
        # 等待连接
        node_socket, node_addr = node.wait_for_connection()
        # 迭代处理每一批数据
        for i in range(iterations):
            logging.info(f"node_{host_ip} 获取来自 node{node_addr} 的连接")
            msg = node.receive_message(node_socket)
            logging.info("msg: %s", msg)
            # 解包消息内容
            _, data, target, start_layer, node_layer, layer_node = msg
            # 计算输出
            data, next_layer, split = calculate_output(model, data, start_layer)
            # 如果不是最后一层就,继续向下一个节点发送数据
            if split + 1 < model_len:
                # 获取下一个节点的IP
                next_client = layer_node[split + 1]
                # 建立连接
                node.connect(next_client, get_client_app_port(next_client, model_name))
                # 封装要发送的数据
                msg = [info, data.cpu(), target.cpu(), next_layer, node_layer, layer_node]
                # 发送
                node.send_message(node.sock, msg)
                print(
                    f"node_{host_ip} send msg to node{config.CLIENTS_LIST[layer_node[split + 1]]}"
                )
            else:
                # 到达最后一层，计算损失和准确率
                loss = F.cross_entropy(data, target)
                acc = calculate_accuracy(data, target)
                loss_list.append(loss)
                acc_list.append(acc)
                print("loss :{:.4f}".format(sum(loss_list) / len(loss_list)))
                print("acc :{:.4f}%".format(100 * sum(acc_list) / len(acc_list)))
                print("")

        # 关闭socket连接
        node_socket.close()
        # 重新初始化节点
        node.__init__(host_ip, host_port)


def from_first(model, node):
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
    for data, target in test_loader:
        # split: 当前节点计算的层
        # next_layer: 下一个权重层
        data, next_layer, split = calculate_output(model, data, start_layer)

        # 获取下一个接收节点的地址，并建立通信
        next_client = config.CLIENTS_LIST[layer_node_indices[split + 1]]
        node.connect(next_client, get_client_app_port(next_client, model_name))

        # 准备发送的消息内容，可能需要包含标签
        msg = [info, data.cpu(), target.cpu(), next_layer, node_layer_indices, layer_node_indices]
        print(
            f"node{host_ip} send msg to node{config.CLIENTS_LIST[layer_node_indices[split + 1]]}"
        )
        node.send_message(node.sock, msg)


def start_inference():
    """
    整个推理过程的入口。
    它初始化模型和节点连接，如果包含第一层，它会加载数据集并计算第一层的输出，然后将结果发送给下一个节点。
    最后，它调用 node_inference 函数开始节点推理过程。
    """
    # 建立连接
    node = NodeEnd(host_ip, host_port)

    # 初始化模型并载入预训练权重
    model = VGG("Client", model_name, len(model_cfg[model_name]) - 1, model_cfg[model_name])
    model.eval()
    model.load_state_dict(torch.load("models/vgg/vgg.pth"))

    node_inference(node, model)


if __name__ == '__main__':
    host_ip = '192.168.215.130'
    host_port = 9001

    loss_list = []
    acc_list = []

    info = "MSG_FROM_NODE_ADDRESS(%s), host= %s" % (host_ip, host_ip)

    model_name = 'VGG'
    model_len = len(model_cfg[model_name])
    node_layer_indices = {}
    layer_node_indices = {}

    start_inference()
