"""
    服务器执行文件，主要负责：
        1、根据客户端的资源使用情况。将模型文件进行分层
"""
# ip及对应节点位序
from Communicator import Communicator
import torch
from models.vgg.VGG import VGG
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import socket
import time
import random
from models.model_struct import model_cfg
from config import CLIENTS_CONFIG, CLIENTS_LIST


# To automatically select segmentation points in the network with minimal performance loss
# The idea is to avoid splitting at critical layers (like the first and last layers) and
# balance the computational load across segments.
# and contains the network's layer configurations
# 随机选择分割点
def select_segmentation_points(model_cfg, num_segments):
    """
    Automatically selects segmentation points for the network.
    :param model_cfg: Configuration of the network models
    :param num_segments: Desired number of segments
    :return: List of segmentation indices
    """
    # 由于num_segments=3，我们需要2个分割点来将网络分割成3部分
    num_split_points = num_segments - 1

    total_layers = len(model_cfg['VGG5'])

    # 生成一个包含所有可用于分割的层的列表，避开第一层和最后一层
    eligible_layers = list(range(1, total_layers - 1))

    # 从可用层中随机选择所需数量的分割点
    segmentation_points = sorted(random.sample(eligible_layers, min(num_split_points, len(eligible_layers))))

    return segmentation_points
    # return sorted(random.sample(segmentation_points, min(num_segments, len(segmentation_points))))

# def choose_split_points(models, num_splits=3):
#     eligible_layers = [i for i, layer in enumerate(models.children()) if isinstance(layer, (nn.Conv3d, nn.Linear))]
#     return sorted(random.sample(eligible_layers, min(num_splits, len(eligible_layers))))
#
# # 创建模型实例
# models = VideoRecognitionCNN()
# # 随机选择分割点
# split_points = choose_split_points(models)


# Segment the network
num_segments = 3  # Number of desired segments
segmentation_points = select_segmentation_points(model_cfg, num_segments)
print('*'*40)
print(segmentation_points)



# Now, segment the network based on the selected points
def segment_network(model_cfg, segmentation_points):
    segments = []
    start = 0
    for point in segmentation_points:
        segments.append(model_cfg['VGG5'][start:point])
        start = point
    segments.append(model_cfg['VGG5'][start:])  # Add the last segment
    return segments

segmented_models = segment_network(model_cfg, segmentation_points)

# Print the layer configurations for each segment and the segmentation points
for i, segment in enumerate(segmented_models):
    print(f"Segment {i+1}:")
    for layer_cfg in segment:
        print(layer_cfg)
    print("\n")  # New line for better readability between segments

# segments now contains the segmented layers of the network
# 在每个节点上计算的第k层
# split_layer = {0: [0, 1], 1: [2, 3], 2: [4, 5, 6]}
# reverse_split_layer = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
first_a = []
second_b = []
third_c = []
for i in range(0,segmentation_points[0]):
    first_a.append(i)
for i in range(segmentation_points[0], segmentation_points[1]):
    second_b.append(i)
for i in range(segmentation_points[1], len(model_cfg['VGG5'])):
    third_c.append(i)

print(first_a,second_b,third_c)
split_layer = {0: first_a, 1: second_b, 2: third_c}
print(split_layer)

reverse_split_layer={}
for i in range(0,len(first_a)):
    reverse_split_layer[i]=0
x = len(first_a)
for i in range(len(first_a)):
    reverse_split_layer[x+i]=1
y = len(first_a)+len(second_b)
for i in range(len(third_c)):
    reverse_split_layer[y+i]=2
print(reverse_split_layer)



host_port = 9001
host_node_num = 0
host_ip = CLIENTS_LIST[host_node_num]

info = "MSG_FROM_NODE(%d), host= %s" % (host_node_num, host_ip)

loss_list = []

model_name = "VGG5"
model_len = len(model_cfg[model_name])

N = 10000 # data length
B = 256 # Batch size

### 假设本节点为节点0
class node_end(Communicator):
    def __init__(self,host_ip,host_port):
        super(node_end, self).__init__(host_ip,host_port)

    def add_addr(self, node_addr, node_port):
        while True:
            try:
                self.sock.connect((node_addr, node_port))
                break  # If the connection is successful, break the loop
            except socket.error as e:
                print(f"Failed to connect to {node_addr}:{node_port}, retrying...")
                time.sleep(1)  # Wait for a while before retrying

#     def send_segmentation_points(self, sock, segmentation_points):
#         msg = ['SegmentationPoints', segmentation_points]
#         self.send_msg(sock, msg)
# # 示例：发送分割点到下一个节点
# node_instance = node_end(host_ip, host_port)
# next_node_sock =node_instance.sock  # 获取下一个节点的套接字
# node_instance.send_segmentation_points(next_node_sock, segmentation_points)


# TODO:理解这个函数
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    #print("preds={}, y.view_as(preds)={}".format(preds, y.view_as(preds)))
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


def node_inference(node, model):
    node.__init__(host_ip,host_port)
    while True:
        global reverse_split_layer,split_layer
        last_send_ips=[]
        iteration = int(N / B)
        node_socket, node_addr = node.wait_for_connection()
        for i in range(iteration):
            print("node_{host_node_num} get connection from node{node_addr}")
            msg = node.recv_msg(node_socket)
            data = msg[1]
            target = msg[2]
            start_layer = msg[3]
            split_layer = msg[4]
            reverse_split_layer = msg[5]
            data, next_layer, split = calculate_output(model, data, start_layer)
            if split + 1 < model_len:
                last_send_ip=CLIENTS_LIST[reverse_split_layer[split + 1]]
                if last_send_ip not in last_send_ips:
                    node.add_addr(last_send_ip, 1998)
                last_send_ips.append(last_send_ip)
                msg = [info, data.cpu(), target.cpu(), next_layer,split_layer,reverse_split_layer]
                node.send_msg(node.sock, msg)
                print(
                    f"node_{host_node_num} send msg to node{CLIENTS_LIST[reverse_split_layer[split + 1]]}"
                )
            else:
                # 到达最后一层，计算损失
                loss = torch.nn.functional.cross_entropy(data, target)
                loss_list.append(loss)
                print("loss :{}".format(sum(loss_list) / len(loss_list)))
                print("")
        node_socket.close()
        node.__init__(host_ip,host_port)

def get_model(model, type, in_channels, out_channels, kernel_size, start_layer):
    # for name, module in models.named_children():
    #   print(f"Name: {name} | Module: {module}")
    # print(models)
    feature_s = []
    dense_s = []
    if type == "M":
        feature_s.append(model.features[start_layer])
        start_layer += 1
    if type == "D":
        ## TODO:denses' modify the start_layer
        dense_s.append(model.denses[start_layer-11])
        start_layer += 1
    if type == "C":
        for i in range(3):
            feature_s.append(model.features[start_layer])
            start_layer += 1
    next_layer = start_layer
    return nn.Sequential(*feature_s), nn.Sequential(*dense_s), next_layer


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
        #print("next_layer", next_layer)
    return data, next_layer, split


def start_inference():
    include_first = True
    node = node_end(host_ip, host_port)

    model = VGG("Client", model_name, 6, model_cfg)
    model.eval()
    model.load_state_dict(torch.load("models/vgg/vgg.pth"))

    # moddel layer Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #print("moddel layer",models)

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

        last_send_ips=[]
        for data, target in test_loader:
            #print(len(data))
            # split:当前节点计算的层
            # next_layer:下一个权重层
            data, next_layer, split = calculate_output(model, data, start_layer)

            # TODO:modify the port
            last_send_ip=CLIENTS_LIST[reverse_split_layer[split + 1]]
            if last_send_ip not in last_send_ips:
                node.add_addr(last_send_ip, 1998)

            last_send_ips.append(last_send_ip)

            # TODO:是否发送labels
            msg = [info, data.cpu(), target.cpu(), next_layer,split_layer,reverse_split_layer]
            print(
                f"node{host_node_num} send msg to node{CLIENTS_LIST[reverse_split_layer[split + 1]]}"
            )
            node.send_msg(node.sock, msg)
            include_first = False
            # print('*' * 40)
        node.sock.close()
    node_inference(node, model)


# start_inference()
if __name__ == '__main__':
    start_inference()