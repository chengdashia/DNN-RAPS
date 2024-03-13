"""
    网络分割策略
"""
import random
from config import CLIENTS_NUMBERS


# 网络分割策略类
class NetworkSegmentationStrategy:
    def __init__(self, model_name, model_cfg):
        """
        Initialize the NetworkSegmentationStrategy class with the network configuration.
        :param model_name: Name of the dnn model
        :param model_cfg: Configuration of the network models
        """
        self.model_name = model_name
        self.model_cfg = model_cfg

    # To automatically select segmentation points in the network with minimal performance loss
    # The idea is to avoid splitting at critical layers (like the first and last layers) and
    # balance the computational load across segments.
    # and contains the network's layer configurations
    def random_select_segmentation_points(self):
        """
        Automatically selects segmentation points for the network.
        :return: List of segmentation indices
        """
        num_split_points = CLIENTS_NUMBERS - 1

        total_layers = len(self.model_cfg[self.model_name])

        # 生成一个包含所有可用于分割的层的列表，避开第一层和最后一层
        eligible_layers = list(range(1, total_layers - 1))

        # 从可用层中随机选择所需数量的分割点
        return sorted(random.sample(eligible_layers, min(num_split_points, len(eligible_layers))))

    def resource_aware_segmentation_points(self, resource_usage):
        """
        根据节点资源使用情况选择分割点
        :param resource_usage: 字典,包含每个节点的CPU、GPU、内存和网络利用率
        :return: 分割点列表
        """
        num_split_points = CLIENTS_NUMBERS - 1
        total_layers = len(self.model_cfg[self.model_name])
        eligible_layers = list(range(1, total_layers - 1))

        # 按CPU、内存和网络利用率对节点进行排序
        sorted_nodes = sorted(resource_usage.items(),
                              key=lambda x: (float(x[1]['cpu'].split(':')[1]), x[1]['memory'], x[1]['network']))

        segmentation_points = []
        current_node_idx = 0

        # 遍历所有层
        for layer_idx in eligible_layers:
            layer_type, _, _, _, _, flops = self.model_cfg[self.model_name][layer_idx]

            # 将CPU利用率从字符串转换为浮点数
            current_node_cpu = float(sorted_nodes[current_node_idx][1]['cpu'].split(':')[1])

            # 如果当前节点资源不足以承载该层,则将其分配给下一个节点
            if layer_type == 'C' and flops > current_node_cpu:
                current_node_idx += 1
                segmentation_points.append(layer_idx)

            # 确保不会超过最大分割点数量
            if len(segmentation_points) >= num_split_points:
                break

        return segmentation_points
