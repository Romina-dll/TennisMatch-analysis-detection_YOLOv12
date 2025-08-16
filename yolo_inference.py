from ultralytics import YOLO

# Initialize the YOLO model with pre-trained weights
model = YOLO('yolo12n.pt')

results = model.track(
    source='input_videos/input_video.mp4',
    save=True,
    # Maintain consistent tracking IDs across frames
    persist=True
    )