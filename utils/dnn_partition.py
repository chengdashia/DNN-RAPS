def segment_network(model_cfg, segmentation_points):
    segments = []
    start = 0
    for point in segmentation_points:
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
