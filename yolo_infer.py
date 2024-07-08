from ultralytics import YOLO, settings

# settings.update({'datasets_dir': '/Users/robin/Git/tennis_system/ultralytics_datasets',
#                  'weights_dir': '/Users/robin/Git/tennis_system/weights',
#                  'runs_dir': '/Users/robin/Git/tennis_system/runs'})
print(settings)
# Load an official or custom model
model = YOLO("./model/yolov8x.pt")

result = model.track("./assert/image.png", save=True)
