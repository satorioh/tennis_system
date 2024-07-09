from utils import save_video, read_video

input_video_path = "./assert/input_video.mp4"
output_video_path = "./output/output_video.mp4"


def main():
    video_frames = read_video(input_video_path)
    save_video(video_frames, output_video_path)


if __name__ == '__main__':
    main()
