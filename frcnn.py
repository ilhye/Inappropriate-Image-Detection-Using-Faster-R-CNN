import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from cocoClass import COCO_CLASSES

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_FONT = ImageFont.load_default()

# COCO Model (for checking 'person')
_WEIGHTS = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
_COCO_MODEL = fasterrcnn_resnet50_fpn(weights=_WEIGHTS).to(DEVICE).eval()
_TF = torchvision.transforms.ToTensor()

# Load your custom-trained model
def get_model(num_classes=3):
    # Load pre-trained Faster R-CNN
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Move model to GPU if available
    model.load_state_dict(torch.load("fasterrcnn_resnet50_epoch_5.pth", map_location=DEVICE))
    return model.to(DEVICE).eval()

_MODEL = get_model()

# Draw bounding boxes with the correct class names and increase image size
# threshol = 0.8
# `prediction` contains:
# - boxes: predicted bounding boxes
# - labels: predicted class labels
# - scores: predicted scores for each box (confidence level)
def draw_boxes(pil_img: Image.Image, score_thresh: float = 0.8) -> Image.Image:
    # Convert image to tensor and add batch dimension
    img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(DEVICE) 

    # Disable gradient computation for inference
    with torch.no_grad():
        pred = _MODEL(img_tensor)[0]

    draw = ImageDraw.Draw(pil_img)
    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < score_thresh:
            continue
        xmin, ymin, xmax, ymax = map(int, box)

        # Get class name from COCO_CLASSES
        class_name = COCO_CLASSES.get(label.item(), "Unknown")

        # Display the image
        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=2)
        draw.text((xmin, ymin), f"{class_name} ({score:.2f})", fill="blue", font=_FONT)

    return pil_img
