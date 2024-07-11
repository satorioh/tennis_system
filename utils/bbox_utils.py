def get_center_of_bbox(bbox):
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_bottom_center_of_bbox(bbox):
    return int((bbox[0] + bbox[2]) / 2), bbox[3]


def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]


def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])


def get_closest_keypoint_index(point, keypoints, ref_keypoints):
    closest_distance_y = float('inf')
    key_point_index = ref_keypoints[0]
    for keypoint_i in ref_keypoints:
        keypoint = keypoints[keypoint_i * 2], keypoints[keypoint_i * 2 + 1]
        distance_y = abs(point[1] - keypoint[1])

        if distance_y < closest_distance_y:
            closest_distance_y = distance_y
            key_point_index = keypoint_i

    return key_point_index


def get_closest_player_by_point(player_bbox, target_point):
    return min(player_bbox.keys(), key=lambda player_id: measure_distance(player_bbox[player_id], target_point))
