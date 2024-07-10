import pickle
import cv2
from os import path
from ultralytics import YOLO
from utils import measure_distance, get_center_of_bbox


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """

        :param court_keypoints: [x1, y1, x2, y2, ...]
        :param player_detections: [{ track_id_1: [x1, y1, x2, y2], track_id_2: [x1, y1, x2, y2] }]
        :return:
        """
        print("Choosing and filtering players...")
        player_dict_from_first_frame = player_detections[0]
        choose_player_ids = self.get_choose_player_ids(court_keypoints, player_dict_from_first_frame)

        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if
                                    track_id in choose_player_ids}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def get_choose_player_ids(self, court_keypoints, player_dict_from_first_frame):
        print("get choose player ids...")
        distances = []
        for track_id, bbox in player_dict_from_first_frame.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            court_keypoints_len = len(court_keypoints)
            for i in range(0, court_keypoints_len, 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        # sort the distances in ascending order
        distances.sort(key=lambda x: x[1])
        return [i[0] for i in distances[:2]]

    def detect_frames(self, frames, stub_path=None):
        player_detections = []

        if stub_path and path.exists(stub_path):
            print("Reading player detections from stub")
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        print("Detecting players...")
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True, classes=[0])[0]

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            player_dict[track_id] = result
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        print("Drawing player bounding boxes...")
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames
