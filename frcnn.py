from gradio_client import file
import torch
import torchvision
import cv2
import numpy as np

from torchvision.transforms import ToTensor, ToPILImage
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from cocoClass import COCO_CLASSES
from purification import adversarial_purification, purification_check, DiffusionModel
from realesrgan_wrapper import load_model as esrgan_load_model, run_sr as esrgan_run_sr

# Load default model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_FONT = ImageFont.load_default()
_WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
_COCO_MODEL = fasterrcnn_resnet50_fpn(weights=_WEIGHTS).to(DEVICE).eval()
_TF = torchvision.transforms.ToTensor()
_diffusion_model = DiffusionModel(model_path="guided_diffusion/models/256x256_diffusion_uncond.pt", image_size=256)
_esrgan_model = esrgan_load_model(device=DEVICE)

def get_model(weights_path="models/fasterrcnn_resnet50_epoch_5.pth", num_classes=11):
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

def draw_boxes(pil_img: Image.Image, score_thresh: float = 0.8) -> Image.Image:
    """ Output media with bounding boxes and their classes
    Args:
        pil_img (PIL.Image): Input image from the routes.py
        score_thresh (float): Threshold to filter boxes based on confidence score
    """
    img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(DEVICE) 

    with torch.no_grad():
        pred = _MODEL(img_tensor)[0]

    predicted_classes = []

    print("Raw labels:", pred["labels"].tolist())
    print("Raw scores:", pred["scores"].tolist())

    draw = ImageDraw.Draw(pil_img)

    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < score_thresh:
            continue
        xmin, ymin, xmax, ymax = map(int, box)

        class_name = COCO_CLASSES.get(label.item(), "Unknown")
        predicted_classes.append(class_name)

        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=2)
        draw.text((xmin, ymin), f"{class_name} ({score:.2f})", fill="blue", font=_FONT)

    return pil_img, predicted_classes

# Image detection
def detect_image(file):
    """Detect objects in an image 
    Args: 
        file: FileStorage (Flask) object, file-like object, string path, or PIL Image. 
                    This is automatically handled by routes
    
    Returns:
        annotated (PIL.Image): Image with bounding boxes drawn
        class_names (list): List of detected class names
    """
    if isinstance(file, Image.Image):
        pil_img = file.convert("RGB")
    elif hasattr(file, 'stream'):
        pil_img = Image.open(file.stream).convert("RGB")
    elif isinstance(file, str):
        pil_img = Image.open(file).convert("RGB")
    else:
        pil_img = Image.open(file).convert("RGB")

    annotated, class_names = draw_boxes(pil_img.copy())
    return annotated, class_names

# def detect_video(input_path, output_path,):
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         raise ValueError("Could not open video")

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, 1, (width, height))

#     frame_idx = 0
#     detections_all = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 1 frame per seconds
#         if frame_idx % int(fps) == 0:
#              # Convert to PIL
#             pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             img_pil = ToTensor()(pil_frame).unsqueeze(0).to(DEVICE)

#             # Purification
#             print("Purifying frame...")
#             purified_frame = adversarial_purification(img_pil, _diffusion_model)
#             print("checking purification...")
#             purified_checked, ok, score = purification_check(img_pil, purified_frame)

#             purified = ToPILImage()(purified_checked.squeeze(0))

#             # Super-Resolution
#             print("super-resolution")
#             sr_frame = esrgan_run_sr(_esrgan_model, purified)
#             print("done sr")

#             # Draw boxes on the processed frame
#             annotated_pil, class_names = draw_boxes(sr_frame)
#             detections_all.extend(class_names)

#             # Convert back to OpenCV
#             annotated_cv2 = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
#             # Ensure correct size
#             annotated_cv2 = cv2.resize(annotated_cv2, (width, height))

#             out.write(annotated_cv2)
#         frame_idx += 1

#     cap.release()
#     out.release()
#     return output_path, detections_all

def detect_video(input_path, output_path):
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only 1 frame per second
        if frame_idx % int(input_fps) == 0:
            # Convert frame to PIL image
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_pil = ToTensor()(pil_frame).unsqueeze(0).to(DEVICE)

            # Purification
            print("Purifying frame...")
            purified_frame = adversarial_purification(img_pil, _diffusion_model)
            print("Checking purification...")
            purified_checked, ok, score = purification_check(img_pil, purified_frame)
            purified = ToPILImage()(purified_checked.squeeze(0))

            # Super-Resolution
            print("Applying super-resolution...")
            sr_frame = esrgan_run_sr(_esrgan_model, purified)
            print("Super-resolution done.")

            # Draw detection boxes
            print("Detecting objects...")
            annotated_pil, class_names = draw_boxes(sr_frame)
            print("Detection done.")
            detections_all.extend(class_names)

            # Convert back to OpenCV
            annotated_cv2 = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
            # Resize to original size just in case
            annotated_cv2 = cv2.resize(annotated_cv2, (width, height))

            # Write frame

            out.write(annotated_cv2)
        
        print(f"frame: {frame_idx}")
        frame_idx += 1

    cap.release()
    out.release()
    print("Done! Video saved as:", output_path)
    return output_path, detections_all