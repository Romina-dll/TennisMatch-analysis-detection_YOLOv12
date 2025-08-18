import cv2  
import pickle  # For serializing/deserializing Python objects
from ultralytics import YOLO 

"""
A class for pradicting the ball in video frames using YOLO object detection.
Features:
    - ball detection and predicting
    - Serialization of detection results
    - Bounding box visualization
"""
class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  

    """
    Detect and predict the ball in a single video frame.
    Args:
        frame (numpy.ndarray): Input video frame in BGR format
    Returns:
        dict: Dictionary mapping track IDs to bounding boxes {track_id[1]: [x1,y1,x2,y2]}
    """
    def detect_frame(self, frame):   
        results = self.model.predict(frame, conf = 0.15)[0]  
        ball_dict = {}  
        
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict

    """
    Process multiple video frames to detect and track the ball.
    Args:
        frames (list[numpy.ndarray]): List of video frames in BGR format. 
        read_from_stub (bool): Whether to load detections from a saved file.
        stub_path (str): Path to the stub file for saving/loading detections.
    Returns:
        list[dict]: List of ball detection dictionaries where each dictionary:
            - Key (int): ball tracking ID (persistent across frames)
            - Value (list[float]): Bounding box coordinates [x1, y1, x2, y2]
    """
    def detect_frames(self, frames , read_from_stub= False , stub_path=None):
        ball_detections = []  # Initialize empty list to store frame-by-frame results
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f :
                ball_detections = pickle.load(f)
                return ball_detections
        for frame in frames:
            # Detect the ball in current frame and get dictionary of:
            # {track_id: [x1, y1, x2, y2]} mappings
            ball_dict = self.detect_frame(frame)
            
            # Append results for current frame to overall detections list
            ball_detections.append(ball_dict)
        if stub_path is not None:
            with open(stub_path , 'wb') as f:
                pickle.dump(ball_detections , f)
        return ball_detections

    """
    Draw bounding boxes and IDs on video frames.
    Args:
        video_frames (list): List of original video frames
        ball_detections (list): List of ball detection dictionaries
    Returns:
        list: List of frames with visualized detections
    """
    def draw_bboxes(self, video_frames, ball_detections):
        
        output_video_frames = []
        
        # Process each frame and its corresponding detections
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                # Draw ball ID text above bounding box
                cv2.putText(frame, f"Ball ID: {track_id}", 
                           (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                           (0, 255, 255), 2)  # yellow text
                
                # Draw bounding box
                cv2.rectangle(frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 255), 2)  # yellow rectangle
                            
            output_video_frames.append(frame)
            
        return output_video_frames