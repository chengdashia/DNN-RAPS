"""
    服务器执行文件，主要负责：
        1、根据客户端的资源使用情况。将模型文件进行分层
        ip及对应节点位序
"""
from node_end import NodeEnd
from models.model_struct import model_cfg
from utils.segment_strategy import NetworkSegmentationStrategy
from utils.resource_utilization import get_all_server_info
from utils.utils import get_client_app_port


def convert_node_layer_indices(node_to_layer):
    """
    将节点层索引字典转换为层节点映射字典
    :param node_to_layer: 节点层索引字典,键为节点IP,值为该节点对应的层索引列表
    :return: 层节点映射字典,键为层索引,值为对应的节点IP
    """
    # 初始化层节点映射字典
    layer_node_mapping = {}

    # 遍历节点层索引字典中的每个节点
    for node_ip, layer_indices in node_to_layer.items():
        # 遍历该节点对应的层索引列表
        for layer_idx in layer_indices:
            # 将层索引和对应的节点IP添加到层节点映射字典
            layer_node_mapping[layer_idx] = node_ip
    return layer_node_mapping


def start():
    # 建立连接
    node = NodeEnd(host_ip, host_port)
    # 准备发送的消息内容
    msg = [info, layer_node_indices]

    node.connect(layer_node_indices[0], get_client_app_port(layer_node_indices[0], model_name))
    node.send_message(node, msg)


if __name__ == '__main__':

    host_port = 9001
    host_ip = '192.168.215.129'

    model_name = "VGG5"

    # 获取所有节点的资源情况
    nodes_resource_infos = get_all_server_info()

    # 分割策略类
    segmentation_strategy = NetworkSegmentationStrategy(model_name, model_cfg)

    # 利用所有节点的资源情况的资源感知分割点方法进行分割
    # segmentation_points:  [2, 4]
    # node_layer_indices:  {'192.168.215.130': [0, 1], '192.168.215.131': [2, 3], '192.168.215.129': [4, 5, 6]}
    segmentation_points, node_layer_indices = (segmentation_strategy
                                               .resource_aware_segmentation_points(nodes_resource_infos))
    print('*' * 40)
    print("resource_aware_segmentation_points  segmentation_points: ", segmentation_points)
    print("resource_aware_segmentation_points  node_layer_indices: ", node_layer_indices)

    # 将节点层索引字典转换为层节点映射字典
    # layer_node_indices:   {
    # 0: '192.168.215.130',
    # 1: '192.168.215.130',
    # 2: '192.168.215.129',
    # 3: '192.168.215.129',
    # 4: '192.168.215.131',
    # 5: '192.168.215.131',
    # 6: '192.168.215.131'
    # }
    layer_node_indices = convert_node_layer_indices(node_layer_indices)

    info = "MSG_FROM_NODE_ADDRESS(%s), host= %s" % (host_ip, host_ip)

    # 开始
    start()
