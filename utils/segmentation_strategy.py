"""
    网络分割策略
"""
import random
from config import CLIENTS_NUMBERS


# 网络分割策略类
class NetworkSegmentationStrategy:
    def __init__(self, model_cfg):
        """
        Initialize the NetworkSegmentationStrategy class with the network configuration.
        :param model_cfg: Configuration of the network models
        """
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
        print(self.model_cfg)

        total_layers = len(self.model_cfg['VGG5'])

        # 生成一个包含所有可用于分割的层的列表，避开第一层和最后一层
        eligible_layers = list(range(1, total_layers - 1))

        # 从可用层中随机选择所需数量的分割点
        return sorted(random.sample(eligible_layers, min(num_split_points, len(eligible_layers))))
