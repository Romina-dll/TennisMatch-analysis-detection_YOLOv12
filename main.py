from utils import read_video, save_video  
from trackers import PlayerTracker , BallTracker  
from court_line_detector import CourtLineDetector

"""
Main execution function for player tracking pipeline.
Workflow:
    1. Reads input video
    2. Initializes player tracker
    3. Detects and tracks players frame-by-frame
    4. Draws visualizations
    5. Saves output video
"""
def main():
    
    
    # Input video configuration
    input_video_path = 'input_videos/input_video.mp4'  # Path to source video
    
    # Step 1: Video Acquisition
    # Read video into list of frames (numpy arrays)
    video_frames = read_video(input_video_path)
    
    # Step 2: Tracker Initialization
    # Load YOLO model with custom player detection weights
    player_tracker = PlayerTracker(model_path='yolo12n.pt')
    #Load YOLOv model with custom ball detection weights
    ball_tracker = BallTracker(model_path='models/ball_detection_best.pt')
    # Initialize court line detector with trained keypoints model
    court_model_path = 'models/keypoints_model_50.pth'
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    # Detect court keypoints from the first frame of the video
    court_keypoints = court_line_detector.predict(video_frames[0])
    
    # Step 3: Player Detection
    # Process all frames to detect and track players , ball
    # Returns list of dictionaries with {player_id: bbox} per frame
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,   # Tells function to load precomputed results
        stub_path='tracker_stubs/player_detection.pkl') # Path to the saved detections file
    # Filter and choose only relevant players based on court keypoints
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints,
        player_detections
    )
    # Returns list of dictionaries with {ball_id: bbox} per frame
    ball_detection = ball_tracker.detect_frames(video_frames ,read_from_stub=True , stub_path='tracker_stubs/ball_detection.pkl')

    # Step 4: Visualization
    # Annotate frames with bounding boxes and player IDs
    output_video_frames = player_tracker.draw_bboxes(
        video_frames, 
        player_detections
    )
    # Annotate frames with bounding boxes and Ball IDs
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detection)
    # Draw court keypoints on video frames
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames , court_keypoints)
    
    # Step 5: Output Generation
    # Save processed video with visualizations
    save_video(output_video_frames, "output_videos/output.avi")

if __name__ == '__main__':
    main()