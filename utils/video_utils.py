import cv2  # OpenCV library for video processing

"""
Reads a video file and extracts all frames as a list of numpy arrays.
Args:
    video_path (str): Path to the input video file
Returns:
    list: List of video frames (numpy arrays)
"""
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)  
    frames = []  
    while True:
        # Read next frame from video
        # ret: Boolean indicating if frame was read successfully
        # frame: The actual frame as numpy array
        ret, frame = cap.read()
        
        if ret:  
            frames.append(frame)  
        else:  # If we've reached end of video or error occurred
            break
    
    cap.release() 
    return frames 


"""
Saves a list of frames as a new video file.
Args:
    output_video_frames (list): List of frames to save (numpy arrays)
    output_video_path (str): Destination path for output video
"""
def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    # Get frame dimensions from first frame (assumes all frames same size)
    frame_height, frame_width = output_video_frames[0].shape[:2]

    # 24: frames per second (adjust based on needs)
    out = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        24,  # Frames per second
        (frame_width, frame_height) 
    )

    for frame in output_video_frames:
        out.write(frame)
    
    out.release()  