import pickle
import cv2
import pandas as pd
from os import path
from ultralytics import YOLO

BALL_PREDICT_CONF = 0.15


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, stub_path=None):
        ball_detections = []

        if stub_path and path.exists(stub_path):
            print("Reading ball detections from stub")
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        print("Detecting ball...")
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=BALL_PREDICT_CONF)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict

    def interpolate_ball_positions(self, ball_positions):
        """
        :param ball_positions: [{}, {1: [895.4949340820312, 613.216064453125, 909.893798828125, 637.4019165039062]}, {}...]
        :return:
        """
        ball_positions = [position.get(1, []) for position in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()  # fill the first missing value
        ball_positions = [{1: result} for result in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def draw_bboxes(self, video_frames, ball_detections):
        print("Drawing ball bounding boxes...")
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames
