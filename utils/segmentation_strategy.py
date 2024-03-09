"""
    分割策略
"""
import random
from config import CLIENTS_NUMBERS


# To automatically select segmentation points in the network with minimal performance loss
# The idea is to avoid splitting at critical layers (like the first and last layers) and
# balance the computational load across segments.
# and contains the network's layer configurations
def select_segmentation_points(model_cfg):
    """
    Automatically selects segmentation points for the network.
    :param model_cfg: Configuration of the network models
    :return: List of segmentation indices
    """
    num_split_points = CLIENTS_NUMBERS - 1

    total_layers = len(model_cfg['VGG5'])

    # 生成一个包含所有可用于分割的层的列表，避开第一层和最后一层
    eligible_layers = list(range(1, total_layers - 1))

    # 从可用层中随机选择所需数量的分割点
    return sorted(random.sample(eligible_layers, min(num_split_points, len(eligible_layers))))
