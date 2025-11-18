"""
===========================================================
Program: FRCNN
Programmer/s: Cristina C. Villasor
Date Written: June 15, 2025
Last Revised: Nov. 18, 2025

Purpose: Handles object detection using Faster R-CNN with custom classes.

Program Fits in the General System Design:
- Detects objects in images/videos after purification and super-resolution
- Provides annotated media with bounding boxes and class names
- Helps determine whether the media contains inappropriate content

Algorithm: 
- Takes input from routes 
- Initialize Faster R-CNN with ResNet-50 backbone and load pre-trained model
- If image, uses detect_image(), then calls draw_boxes() to get detected classes and annotated images
- If video, uses detect_video() on every 10th frame. Each frame undergoes purification and super-resolution before the object detection. After processing all frames, resources are released and final output is saved in the annotated folder.
- Each detection will undergo VQA, then a final score will be created.

Data Structures and Controls: 
- Uses a list for storing classes
- Uses a while loop for processing frames
===========================================================
"""
import torch
import cv2
import numpy as np
import subprocess

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from cocoClass import COCO_CLASSES
from purify.purification import Purifier
from purify.realesrgan import RealESRGANWrapper
from qa import vqa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
_FONT = ImageFont.load_default()                                      # Default font for drawing 

def get_model(weights_path="models/content_mod.pth", num_classes=11):
    """ Load custom classes
    Args:
        weights_path (str): Path to model weights
        num_classes (int): Number of classes including background or 0 index
    Returns:
        model (torch.nn.Module): Loaded model in eval mode
    """
    try:
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    
        return model.to(DEVICE).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

_MODEL = get_model() # Load the model

def draw_boxes(pil_img: Image.Image, score_thresh: float = 0.7) -> Image.Image:
    """ Output media with bounding boxes and their classes
    Args:
        pil_img (PIL.Image): Input image from the routes.py
        score_thresh (float): Threshold to filter boxes based on confidence score
    Returns:
        pil_img (PIL.Image): Image with bounding boxes drawn
        predicted_classes (list): List of detected class names
    """
    # Convert PIL to tensor and add batch dimension
    img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad(): # Disable gradient calculation for inference
        pred = _MODEL(img_tensor)[0]

    predicted_classes = []
    predicted_scores = []

    draw = ImageDraw.Draw(pil_img) # For drawing boxes

    score_pred = 0

    # Draw boxes for each detected object
    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < score_thresh:
            continue
        xmin, ymin, xmax, ymax = map(int, box) # Convert to int for drawing

        class_name = COCO_CLASSES.get(label.item(), "Unknown")
        predicted_classes.append(class_name)
        predicted_scores.append(float(score))

        print(f"Detected classes: {predicted_classes}, Scores: {predicted_scores}")


        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=2)
        draw.text((xmin, ymin), f"{class_name} ({score:.2f})", fill="blue", font=_FONT)

    return pil_img, predicted_classes, predicted_scores

def detect_image(input_img):
    """Detect objects in an image 
    Args: 
        input_img (PIL.Image): Input image in PIL format
    
    Returns:
        annotated_img (PIL.Image): Image with bounding boxes drawn
        class_names (list): List of detected class names
        total_score (float): Final score after VQA and object detection
    """
    annotated_img, class_names, scores = draw_boxes(input_img.copy())     # Object detection
    answers, confidences = vqa.get_answer(input_img)                      # VQA
    total_score = vqa.decision(class_names, answers, confidences, scores) # Compute final score
    
    return annotated_img, class_names, total_score

def mp4_to_h264(input_path):
    """Re-encode video from mp4 to h264 making it compatible playing in browsers
    Args: 
        input_path (str): Path of video to convert
    Returns:
        h264_path (str): Path of converted video
    """
    h264_path =input_path.replace(".mp4", "_h264.mp4") # Replace .mp4 with _h264.mp4

    try:
        subprocess.run([                                   # Run ffmpeg command
            "ffmpeg", "-y", "-i", input_path,              # -y: overwrite files, i: input file
            "-c:v", "libx264",                             # Re-encode mp4 to h264
            "-c:a", "aac",                                 # Re-encode audio to aac
            h264_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        return h264_path
    except Exception as e:
        print(f"Error converting video: {e}")
        raise

def detect_video(input_path, output_path):
    """Detect objects in a video at 1 FPS
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to save the annotated video
    Returns:
        output_path (str): Path to the saved annotated video
        detections_all (list): List of all detected class names in the video
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")

    # Get input video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, input_fps, (width, height))
   
    frame_idx = 0          # Frame counter
    detected_classes = []  # Store detected classes
    all_scores = []        # Store all scores from detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every frame but only annotate every 10th frame for performance
        if frame_idx % 10 == 0:
            # Convert frame to PIL image
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Purification
            purified = Purifier.process(pil_frame)

            # Super-resolution
            sr_frame = RealESRGANWrapper.enhance(purified)

            # Object detection
            annotated_frame, class_names, scores = draw_boxes(sr_frame)
            print("Classes name", class_names)
            detected_classes.extend(class_names) # Append detected classes

            # VQA
            answers, confidences = vqa.get_answer(sr_frame)

            # Compute final score
            frame_score = vqa.decision(detected_classes, answers, confidences, scores)
            all_scores.append(frame_score)

            # Convert back to OpenCV and write annotated frame
            annotated_cv2 = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
            annotated_cv2 = cv2.resize(annotated_cv2, (width, height))

            # Write annotated frame to output video
            out.write(annotated_cv2)
        else:
            # Write original frame (no annotation) to maintain video timing
            out.write(frame)

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    output_path = mp4_to_h264(output_path) 

    total_score = max(all_scores) if all_scores else 0.0
    print(f"Total score: {total_score}")

    return output_path, detected_classes, total_score