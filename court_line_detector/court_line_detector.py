import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

"""
Detects court lines/keypoints in images using a pre-trained ResNet50 model.
Features:
    - Load a fine-tuned ResNet50 model for keypoint regression
    - Predict keypoints on an image
    - Draw keypoints on an image or video frames
"""
class CourtLineDetector:
    """
    Initialize the detector with a pre-trained model.
    Args:
        model_path (str): Path to the saved model weights (.pth file)
    Note: The Tennis Court Keypoint Dataset (tennis_court_det_dataset.zip) is used with a ResNet50 pre-trained model,
          where the final layer is replaced to predict 14×2 outputs (x, y coordinates for 14 keypoints). 
          The model is trained using this dataset, and the trained weights are saved to keypoints_model_50.pth.
    """
    def __init__(self , model_path):
        self.model = models.resnet50(pretrained = True)
        # Replace the final fully-connected layer for 14 keypoints (x, y) => 14*2 outputs
        self.model.fc = torch.nn.Linear(self.model.fc.in_features , 14*2)
        # Load the trained weights from file
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),   # Convert NumPy image to PIL Image
            transforms.Resize((224,224)),    # Resize to 224x224 for ResNet input
            transforms.ToTensor(),   # Convert image to PyTorch tensor
            transforms.Normalize(mean=[0.485,0.456,0.406] , std=[0.229,0.224,0.225])
        ])

    """
    Predict 14 court keypoints for a given image.
    Args:
        image (numpy.ndarray): Input BGR image
    Returns:
        numpy.ndarray: Flattened array of keypoints [x1, y1, x2, y2, ..., x14, y14] in original image scale
    """
    def predict(self , image):
        image_rgb = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
        # Apply preprocessing transforms and add batch dimension
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs= self.model(image_tensor)
        # Convert tensor to NumPy array and flatten
        keypoints = outputs.squeeze().cpu().numpy()
        original_height , original_width = image.shape[:2]
        keypoints[::2] *= original_width / 224.0
        keypoints[1::2] *= original_height / 224.0
        return keypoints
        
    """
    Draw keypoints on a single image.
    Args:
        image (numpy.ndarray): Image to draw on
        keypoints (numpy.ndarray): Flattened array of keypoints [x1, y1, ..., x14, y14]
    Returns:
        numpy.ndarray: Image with keypoints and labels drawn
    """
    def draw_keypoints(self,image,keypoints):
        for i in range (0 , len(keypoints) , 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(image , str( i//2), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX , 0.5,(0,0,255),2)
            cv2.circle(image,(x,y),5,(0,0,255), -1)
        return image
        
    """
    Draw keypoints on all frames of a video.
    Args:
        video_frames (list): List of video frames (numpy arrays)
        keypoints (numpy.ndarray): Flattened array of keypoints to draw on each frame
    Returns:
        list: List of frames with keypoints drawn
    """
    def draw_keypoints_on_video(self, video_frames , keypoints):
        output_videl_frames = []
        for frame in video_frames :
            frame = self.draw_keypoints(frame,keypoints)
            output_videl_frames.append(frame)
        return output_videl_frames