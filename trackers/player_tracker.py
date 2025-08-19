import cv2  
import pickle  # For serializing/deserializing Python objects
from ultralytics import YOLO 
from utils import measure_distance , get_center_bbox

"""
A class for tracking players in video frames using YOLO object detection.
Features:
    - Player detection and tracking
    - Serialization of detection results
    - Bounding box visualization
"""
class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  

    """
    Select the two closest players to the court keypoints and filter detections.
    Args:
        court_keypoints (list): List of keypoints representing positions on the court.
        player_detections (list[dict]): List of player detection dictionaries for each frame.
    Returns:
        list[dict]: Filtered player detections for each frame, keeping only the chosen players.
    """
    def choose_and_filter_players(self , court_keypoints , player_detections):
        player_detection_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints,player_detection_first_frame)
        filtered_player_detections = []
        for player_det in player_detections:
            filter_player_dict = {track_id : bbox for track_id, bbox in player_det.items() if track_id in chosen_player}
            filtered_player_detections.append(filter_player_dict)
        return filtered_player_detections

    """
    Determine the two players closest to the court keypoints.
    Args:
        court_keypoints (list): List of keypoints representing positions on the court.
        player_dict (dict): Dictionary of detected players in a frame {track_id: bbox}.
    Returns:
        list: List of two chosen player track IDs closest to the court keypoints.
    """
    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            # Get the center coordinates of the player's bounding box
            player_center = get_center_bbox(bbox)
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance: # Update minimum distance if this keypoint is closer
                    min_distance = distance
            # Store the player's track_id with its closest distance to any court keypoint
            distances.append((track_id, min_distance))
        #Sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        #Choose the first two tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    """
    Detect and track players in a single video frame.
    Args:
        frame (numpy.ndarray): Input video frame in BGR format
    Returns:
        dict: Dictionary mapping track IDs to bounding boxes {track_id: [x1,y1,x2,y2]}
    """
    def detect_frame(self, frame):   
        results = self.model.track(frame, persist=True)[0]  
        class_names = results.names  
        player_dict = {}  # Dictionary to store player detections
        
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])  
            result = box.xyxy.tolist()[0]  
            class_ids = box.cls.tolist()[0]  
            det_class_names = class_names[class_ids]  
            
            # Only store detections classified as "person"
            if det_class_names == "person":
                player_dict[track_id] = result
                
        return player_dict

    """
    Process multiple video frames to detect and track players.
    Args:
        frames (list[numpy.ndarray]): List of video frames in BGR format. 
        read_from_stub (bool): Whether to load detections from a saved file.
        stub_path (str): Path to the stub file for saving/loading detections.
    Returns:
        list[dict]: List of player detection dictionaries where each dictionary:
            - Key (int): Player tracking ID (persistent across frames)
            - Value (list[float]): Bounding box coordinates [x1, y1, x2, y2]
    """
    def detect_frames(self, frames , read_from_stub= False , stub_path=None):
        player_detections = []  # Initialize empty list to store frame-by-frame results
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f :
                player_detections = pickle.load(f)
                return player_detections
        for frame in frames:
            # Detect players in current frame and get dictionary of:
            # {track_id: [x1, y1, x2, y2]} mappings
            player_dict = self.detect_frame(frame)
            
            # Append results for current frame to overall detections list
            player_detections.append(player_dict)
        if stub_path is not None:
            with open(stub_path , 'wb') as f:
                pickle.dump(player_detections , f)
        return player_detections

    """
    Draw bounding boxes and IDs on video frames.
    Args:
        video_frames (list): List of original video frames
        player_detections (list): List of player detection dictionaries
    Returns:
        list: List of frames with visualized detections
    """
    def draw_bboxes(self, video_frames, player_detections):
        
        output_video_frames = []
        
        # Process each frame and its corresponding detections
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                # Draw player ID text above bounding box
                cv2.putText(frame, f"Player ID: {track_id}", 
                           (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                           (0, 0, 255), 2)  # Red text
                
                # Draw bounding box
                cv2.rectangle(frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 0, 255), 2)  # Red rectangle
                            
            output_video_frames.append(frame)
            
        return output_video_frames