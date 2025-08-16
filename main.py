from utils import read_video, save_video  # Custom video I/O utilities
from trackers import PlayerTracker  # Player detection and tracking module

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
    
    # Step 3: Player Detection
    # Process all frames to detect and track players
    # Returns list of dictionaries with {player_id: bbox} per frame
    player_detections = player_tracker.detect_frames(video_frames)
    
    # Step 4: Visualization
    # Annotate frames with bounding boxes and player IDs
    output_video_frames = player_tracker.draw_bboxes(
        video_frames, 
        player_detections
    )
    
    # Step 5: Output Generation
    # Save processed video with visualizations
    save_video(output_video_frames, "output_videos/output.avi")

if __name__ == '__main__':
    main()