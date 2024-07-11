def convert_pixel_distance_to_meters(pixel_distance, ref_height_in_meter, ref_height_in_pixel):
    return pixel_distance * (ref_height_in_meter / ref_height_in_pixel)


def convert_meters_to_pixel_distance(meters, ref_height_in_meter, ref_height_in_pixel):
    return meters * (ref_height_in_pixel / ref_height_in_meter)


def get_key_point_by_index(index, keypoints):
    return keypoints[index * 2], keypoints[index * 2 + 1]
