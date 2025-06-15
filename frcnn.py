import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from cocoNames import COCO_NAMES 
from PIL import Image, ImageDraw, ImageFont

# Constants
_FONT = ImageFont.load_default()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
_MODEL = fasterrcnn_resnet50_fpn(weights=_WEIGHTS).to(DEVICE).eval()
_TF = T.Compose([T.ToTensor()])

# Confidence score is 0.8

# Check if image contains person
def contains_label(pil_img: Image.Image, label_to_find: str, score_thresh: float = 0.8) -> bool:
    assert label_to_find in COCO_NAMES, f"{label_to_find=} not in COCO"
    img_tensor = _TF(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = _MODEL(img_tensor)[0]

    for lbl, sc in zip(out["labels"].cpu(), out["scores"].cpu()):
        if sc >= score_thresh and COCO_NAMES[lbl] == label_to_find:
            return True
    return False

# Annotate image
def annotate(pil_img: Image.Image,
             score_thresh: float = 0.8) -> Image.Image:
    img_tensor = _TF(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = _MODEL(img_tensor)[0]

    draw = ImageDraw.Draw(pil_img)
    for box, lbl, sc in zip(out["boxes"], out["labels"], out["scores"]):
        if sc < score_thresh:
            continue
        xmin, ymin, xmax, ymax = map(int, box)
        cls_name = COCO_NAMES[lbl]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), f"{cls_name}:{sc:.2f}", fill="red", font=_FONT)

    return pil_img