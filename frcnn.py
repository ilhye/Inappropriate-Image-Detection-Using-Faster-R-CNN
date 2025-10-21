import torch
import cv2
import numpy as np

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from cocoClass import COCO_CLASSES
from purify.purification import Purifier
from purify.realesrgan import RealESRGANWrapper
from qa import vqa

# Load default model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_FONT = ImageFont.load_default()

def get_model(weights_path="models/content_mod.pth", num_classes=11):
    """ Load custom classes
    Args:
        weigth_path (str): Path to model weights
        num_classes (int): Number of classes including background or 0 index
    """
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return model.to(DEVICE).eval()

_MODEL = get_model()

def draw_boxes(pil_img: Image.Image, score_thresh: float = 0.7) -> Image.Image:
    """ Output media with bounding boxes and their classes
    Args:
        pil_img (PIL.Image): Input image from the routes.py
        score_thresh (float): Threshold to filter boxes based on confidence score
    Returns:
        pil_img (PIL.Image): Image with bounding boxes drawn
        predicted_classes (list): List of detected class names
    """
    img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(DEVICE) 

    with torch.no_grad():
        pred = _MODEL(img_tensor)[0]

    predicted_classes = []

    print("Raw labels:", pred["labels"].tolist())
    print("Raw scores:", pred["scores"].tolist())

    # draw = ImageDraw.Draw(pil_img)

    score_pred = 0

    # Draw boxes and labels
    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < score_thresh:
            continue
        xmin, ymin, xmax, ymax = map(int, box) # Convert to int for drawing

        # Display class name
        class_name = COCO_CLASSES.get(label.item(), "Unknown")
        predicted_classes.append(class_name)
        print(f"Detected classes: {predicted_classes}, Scores: {score}")

        score_pred = score

    return pil_img, predicted_classes, score_pred

def detect_image(input_img):
    """Detect objects in an image 
    Args: 
        input_img (PIL.Image): Input image in PIL format
    
    Returns:
        annotated (PIL.Image): Image with bounding boxes drawn
        class_names (list): List of detected class names
    """
    annotated, class_names, scores = draw_boxes(input_img.copy())

    # VQA
    answers, confidences = vqa.get_answer(input_img)
    total_score = vqa.decision(class_names, answers, confidences, scores)
    
    return annotated, class_names, total_score

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Output video at 1 FPS (1 frame per second)
    out = cv2.VideoWriter(output_path, fourcc, 1, (width, height))

    frame_idx = 0
    detections_all = []

    # Process video frame by frame at 1 FPS
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only 10 frames per second
        if frame_idx % 10 == 0:
            # Convert frame to PIL image
            print(type(frame))
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Purification
            print("Purifying frame...")
            purified = Purifier.process(pil_frame)
            print("Purification done.")

            # Super-Resolution
            print("Applying super-resolution...")
            sr_frame = RealESRGANWrapper.enhance(purified)
            print("Super-resolution done.")

            # Draw detection boxes
            print("Detecting objects...")
            annotated_pil, class_names, scores = draw_boxes(sr_frame)
            print("Detection done.")
            detections_all.extend(class_names)

            # VQA
            answers, confidences = vqa.get_answer(sr_frame)
            total_score = vqa.decision(class_names, answers, confidences, scores)

            # Convert back to OpenCV
            annotated_cv2 = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
            # Resize to original size just in case
            annotated_cv2 = cv2.resize(annotated_cv2, (width, height))

            # Write frame
            out.write(annotated_cv2)

    # Release resources
    cap.release()
    out.release()
    print("Done! Video saved as:", output_path)
    return detections_all, total_score