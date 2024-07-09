from ultralytics import YOLO, settings

# settings.update({'datasets_dir': '/Users/robin/Git/tennis_system/ultralytics_datasets',
#                  'weights_dir': '/Users/robin/Git/tennis_system/weights',
#                  'runs_dir': '/Users/robin/Git/tennis_system/runs'})
print(settings)
video_path = "./assert/input_video.mp4"
image_path = "./assert/image.png"
# Load an official or custom model
model = YOLO("./model/yolo5_tennis_ball.pt")

result = model.track(video_path, conf=0.2, save=True)
