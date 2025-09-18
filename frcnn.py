import torch
import torchvision
import cv2
import numpy as np

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from cocoClass import COCO_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_FONT = ImageFont.load_default()

_WEIGHTS = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
_COCO_MODEL = fasterrcnn_resnet50_fpn(weights=_WEIGHTS).to(DEVICE).eval()
_TF = torchvision.transforms.ToTensor()

def get_model(weights_path="trained-model/fasterrcnn_resnet50_epoch_5.pth", num_classes=3):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return model.to(DEVICE).eval()

_MODEL = get_model()

def draw_boxes(pil_img: Image.Image, score_thresh: float = 0.8) -> Image.Image:
    img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(DEVICE) 

    with torch.no_grad():
        pred = _MODEL(img_tensor)[0]

    predicted_classes = []

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
    pil_img = Image.open(file.stream).convert("RGB")
    annotated, class_names = draw_boxes(pil_img.copy())
    return annotated, class_names

# Video detection
def detect_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detections_all = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated_pil, class_names = draw_boxes(pil_frame)
        detections_all.extend(class_names)

        annotated_cv2 = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
        out.write(annotated_cv2)

    cap.release()
    out.release()
    return output_path, detections_all
