from utils.resource_utilization import get_all_server_info
from utils.segment_strategy import NetworkSegmentationStrategy
from models.model_struct import model_cfg
import logging


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
    print("Model segments:")
    for i, segment in enumerate(segments):
        print(f"Segment {i + 1}:")
        for layer_cfg in segment:
            print(layer_cfg)
        # New line for better readability between segments
        print("\n")
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


if __name__ == '__main__':
    model_name = 'VGG5'

    # 获取所有节点的资源情况
    nodes_resource_infos = get_all_server_info()

    print(nodes_resource_infos)

    # 根据不同的分割策略,选取分割点
    segmentation_strategy = NetworkSegmentationStrategy(model_name, model_cfg)
    segmentation_points = segmentation_strategy.random_select_segmentation_points()
    print('*' * 40)
    print("random_select_segmentation_points  segmentation_points: ", segmentation_points)

    segmentation_points, node_layer_indices = segmentation_strategy.resource_aware_segmentation_points(nodes_resource_infos)
    print('*' * 40)
    print("resource_aware_segmentation_points  segmentation_points: ", segmentation_points)
    print("resource_aware_segmentation_points  node_layer_indices: ", node_layer_indices)

    segments = segment_network(model_name, segmentation_points)
    print("segments  : ", segments)





