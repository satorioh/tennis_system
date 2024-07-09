import pickle
from os import path

import cv2

from ultralytics import YOLO


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, stub_path=None):
        player_detections = []

        if stub_path and path.exists(stub_path):
            print("Reading from stub")
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
