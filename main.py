from utils import save_video, read_video
from trackers import PlayerTracker

input_video_path = "./assert/input_video.mp4"
output_video_path = "./output/output_video.mp4"
player_detect_model_path = "./model/yolov8x.pt"
ball_detect_model_path = "./model/yolo5_tennis_ball.pt"
player_stub_path = "./tracker_stubs/player_detections.pkl"


def main():
    video_frames = read_video(input_video_path)
    player_tracker = PlayerTracker(model_path=player_detect_model_path)
    player_detections = player_tracker.detect_frames(video_frames, stub_path=player_stub_path)
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    save_video(output_video_frames, output_video_path)


if __name__ == '__main__':
    main()
