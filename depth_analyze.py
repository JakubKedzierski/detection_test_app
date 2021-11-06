from enum import Enum, auto
import numpy as np

class DepthAnalyze:
    median = auto()
    mean = auto()

def analyzeDepth(image_depth, output, analyze_pipeline=DepthAnalyze.median):
    image_depth_orginal = image_depth.as_depth_frame()
    bbox_to_remove = []

    for bbox in output.bboxes:
        point_LD = (bbox[1], bbox[0])  # left down
        point_RU = (bbox[3], bbox[2])  # right upper
        distance = 0

        clipped_size = 20
        width = bbox[3] - bbox[1] - clipped_size
        height = bbox[2] - bbox[0] - clipped_size

        if width <= 0:
            width = 1
        if height <= 0:
            height = 1

        depth_array = []
        it_x = -1
        it_y = -1

        for i in range(int(bbox[1]) + int(clipped_size / 2), int(bbox[3]) - int(clipped_size / 2)):
            it_x = it_x + 1
            it_y = -1
            for j in range(int(bbox[0]) + int(clipped_size / 2), int(bbox[2]) - int(clipped_size / 2)):
                it_y = it_y + 1
                depth = image_depth_orginal.get_distance(i, j)
                if depth != 0:
                    depth_array.append(depth)

                distance = distance + depth

        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        mean_distance = distance / area
        median_distance = np.median(np.array(depth_array))

        distance = 0
        if analyze_pipeline is DepthAnalyze.mean:
            distance = mean_distance
        else:
            distance = median_distance

        max_distance = 0.55
        if distance > max_distance:
            bbox_to_remove.append(bbox)

    for box in bbox_to_remove:
        result = np.where((output.bboxes == box).all(axis=1))
        output.bboxes = np.delete(output.bboxes, result, 0)
        output.class_ids = np.delete(output.class_ids, result)
        if len(output.masks) > 0:
            output.masks = np.delete(output.masks, result, 2)
        if len(output.scores) > 0:
            output.scores = np.delete(output.scores, result)

    return output
