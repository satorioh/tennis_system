import cv2

from utils import save_video, read_video
from trackers import PlayerTracker, BallTracker
from court_detector import CourtKeyPointsDetector

need_output = True
input_video_path = "./assert/input_video.mp4"
output_video_path = "./output/output_video.mp4"

# Model Paths
player_detect_model_path = "./model/yolov8x.pt"
ball_detect_model_path = "./model/yolo5_tennis_ball.pt"
keypoints_model_path = "./model/keypoints_model.pth"

# Stub Paths
player_stub_path = "./tracker_stubs/player_detections.pkl"
ball_stub_path = "./tracker_stubs/ball_detections.pkl"


def draw_frame_number(video_frames):
    for i, frame in enumerate(video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def main():
    video_frames = read_video(input_video_path)
    player_tracker = PlayerTracker(model_path=player_detect_model_path)
    ball_tracker = BallTracker(model_path=ball_detect_model_path)
    court_keypoints_detector = CourtKeyPointsDetector(keypoints_model_path)

    # ----------------------Detection----------------------#
    # Detect Players and Ball
    player_detections = player_tracker.detect_frames(video_frames, stub_path=player_stub_path)
    ball_detections = ball_tracker.detect_frames(video_frames, stub_path=ball_stub_path)

    # Interpolate Ball Positions
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect Court Key Points
    court_keypoints = court_keypoints_detector.predict(video_frames[0])

    # Choose and Filter Player detections
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # ----------------------Drawing----------------------#
    # Draw Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Draw Court Key Points
    output_video_frames = court_keypoints_detector.draw_keypoints_on_video_frames(output_video_frames, court_keypoints)

    # Draw Frame Number
    draw_frame_number(output_video_frames)

    # ----------------------Save Video----------------------#
    if need_output:
        save_video(output_video_frames, output_video_path)


if __name__ == '__main__':
    main()
