"""
===========================================================
Program: FRCNN
Programmer/s: Cristina C. Villasor
Date Written: June 15, 2025
Last Revised: Oct. 21, 2025

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
import os 

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
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return model.to(DEVICE).eval()

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
    img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(DEVICE)  # Convert PIL to tensor and add batch dimension

    with torch.no_grad(): # Disable gradient calculation for inference
        pred = _MODEL(img_tensor)[0]

    predicted_classes = []

    draw = ImageDraw.Draw(pil_img) # For drawing boxes

    score_pred = 0

    # Draw boxes for each detected object
    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < score_thresh:
            continue
        xmin, ymin, xmax, ymax = map(int, box) # Convert to int for drawing

        # Display class name
        class_name = COCO_CLASSES.get(label.item(), "Unknown")
        predicted_classes.append(class_name)
        print(f"Detected classes: {predicted_classes}, Scores: {score}")

        score_pred = score

        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=2)
        draw.text((xmin, ymin), f"{class_name} ({score:.2f})", fill="blue", font=_FONT)

    return pil_img, predicted_classes, score_pred

def detect_image(input_img):
    """Detect objects in an image 
    Args: 
        input_img (PIL.Image): Input image in PIL format
    
    Returns:
        annotated_img (PIL.Image): Image with bounding boxes drawn
        class_names (list): List of detected class names
        total_score (float): Final score after VQA and object detection
    """
    annotated_img, class_names, scores = draw_boxes(input_img.copy())
    answers, confidences = vqa.get_answer(input_img)
    total_score = vqa.decision(class_names, answers, confidences, scores)
    
    return annotated_img, class_names, total_score

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
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Try different codecs for better compatibility
    for codec in ['mp4v', 'avc1', 'H264']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, input_fps, (width, height))
        if out.isOpened():
            print(f"Using codec: {codec}")
            break
    else:
        print("ERROR: Could not initialize video writer with any codec")
        cap.release()
        return output_path, [], 0.0

    frame_idx = 0
    detected_classes = []
    all_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every frame but only annotate every 10th frame for performance
        if frame_idx % 10 == 0:
            # Convert frame to PIL image
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            purified = Purifier.process(pil_frame)                    # Purification
            sr_frame = RealESRGANWrapper.enhance(purified)            # Super-resolution
            annotated_frame, class_names, scores = draw_boxes(sr_frame) # Object detection
            detected_classes.extend(class_names)                      # Append detected classes
            print("Detected classes so far:", detected_classes)
            answers, confidences = vqa.get_answer(sr_frame)           # VQA
            frame_score = vqa.decision(class_names, answers, confidences, scores) # Final score
            all_scores.append(frame_score)

            # Convert back to OpenCV and write annotated frame
            annotated_cv2 = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
            annotated_cv2 = cv2.resize(annotated_cv2, (width, height))
            out.write(annotated_cv2)
        else:
            # Write original frame (no annotation) to maintain video timing
            out.write(frame)

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    # Verify the output video was created and has content
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Video file size: {file_size} bytes")
        if file_size == 0:
            print("ERROR: Video file is empty!")
        else:
            print(f"Video file created successfully: {output_path}")
    else:
        print(f"ERROR: Video file was not created: {output_path}")

    print("Done! Video saved as:", output_path)

    # Reopen the output video to check frame count
    out_cap = cv2.VideoCapture(output_path)
    out_frame_count = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = out_cap.get(cv2.CAP_PROP_FPS)
    out_duration = out_frame_count / out_fps if out_fps > 0 else 0
    out_cap.release()

    print(f"Output video frames: {out_frame_count}, FPS: {out_fps}, Duration: {out_duration:.2f} seconds")
    total_score = (sum(all_scores) / len(all_scores)) if all_scores else 0.0
    print("Final total score:", total_score)
    return output_path, detected_classes, total_score