from utils.resource_utilization import get_all_server_info
from utils.segmentation_strategy import NetworkSegmentationStrategy
from models.model_struct import model_cfg
import config
if __name__ == '__main__':
    model_name = 'VGG5'
    # 获取所有节点的资源情况
    nodes_resource_infos = get_all_server_info(config.server_list)

    # 根据不同的分割策略,选取分割点
    segmentation_strategy = NetworkSegmentationStrategy(model_name, model_cfg)
    segmentation_points = segmentation_strategy.random_select_segmentation_points()
    print('*' * 40)
    print("segmentation_points: ", segmentation_points)


    segmentation_points = segmentation_strategy.resource_aware_segmentation_points(nodes_resource_infos)
    print('*' * 40)
    print("segmentation_points: ", segmentation_points)


