# 在每个节点上计算的第k层
# split_layer = {0: [0, 1], 1: [2, 3], 2: [4, 5, 6]}
# reverse_split_layer = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
"""
假设segmented_models已经定义，它是一个列表，其中每个元素表示一个分段的模型。
初始化两个空字典split_layer和reverse_split_layer，用于存储分层信息。
使用start_index变量来跟踪当前分段模型的起始索引。
使用enumerate()函数遍历segmented_models中的每个分段模型及其对应的索引i。
对于每个分段模型，使用range()函数生成一个从start_index开始，长度为当前分段模型长度的索引列表layer_indices。
将layer_indices存储在split_layer字典中，以i作为键。
使用内部循环遍历layer_indices中的每个索引j，并将其对应的分段索引i存储在reverse_split_layer字典中，以j作为键。
更新start_index，将其增加当前分段模型的长度，以便下一次迭代时正确跟踪起始索引。
打印split_layer和reverse_split_layer字典。
这样，你就可以根据segmented_models的长度来自动获取split_layer和reverse_split_layer字典了。split_layer字典将分段索引映射到对应的层索引列表，而reverse_split_layer字典将层索引映射到对应的分段索引。
"""
def segmented_index(segmented_models):
    split_layer = {}
    reverse_split_layer = {}
    start_index = 0
    for i, model in enumerate(segmented_models):
        layer_indices = list(range(start_index, start_index + len(model)))
        split_layer[i] = layer_indices
        for j in layer_indices:
            reverse_split_layer[j] = i
        start_index += len(model)
    print(split_layer)
    print(reverse_split_layer)
    return split_layer, reverse_split_layer

