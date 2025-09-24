import os
import numpy as np
import torch

from flask import Blueprint, render_template, request, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from cocoClass import COCO_CLASSES
from frcnn import detect_image, detect_video
from purification import adversarial_anti_aliasing, adversarial_purification, DiffusionModel
from realesrgan_wrapper import load_model as esrgan_load_model, run_sr as esrgan_run_sr

bp = Blueprint("routes", __name__)

# folders
UPLOAD_IMG_FOLDER = os.path.join("static", "uploads")
ANNOT_IMG_FOLDER = os.path.join("static", "annotated")
os.makedirs(UPLOAD_IMG_FOLDER, exist_ok=True)
os.makedirs(ANNOT_IMG_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_diffusion_model = DiffusionModel(model_path="guided_diffusion/models/256x256_diffusion.pt", image_size=256)
_esrgan_model = esrgan_load_model(device=DEVICE)

# Flask form
class CreatePost(FlaskForm):
    """Form for uploading media"""
    uploadImg = FileField(
        "Upload File",
        validators=[FileRequired(), FileAllowed(["jpg", "jpeg", "png", "mp4", "avi", "mov"], "Images/Videos only")]
    )
    submit = SubmitField("Submit")

def run_purification(image_pil: Image.Image) -> Image.Image:
    """Guided diffusion purification 
    Args:
        image_pil (PIL.Image): Input image
        
    Returns:
        PIL.Image: Purified image
    """
    img_t = ToTensor()(image_pil).unsqueeze(0).to(DEVICE)
    aa = adversarial_anti_aliasing(img_t, sigma=1.0)
    purified_t = adversarial_purification(aa, _diffusion_model)
    return ToPILImage()(purified_t.squeeze(0))

def run_realesrgan(image_pil: Image.Image) -> Image.Image:
    """Use Real-ESRGAN for super-resolution
    Args:
        image_pil (PIL.Image): Input image
    
    Returns:
        PIL.Image: Enhanced image
    """
    return esrgan_run_sr(_esrgan_model, image_pil)

@bp.route("/", methods=["GET", "POST"])
def content_moderation():
    """Main route for content moderation. This handles image/video.
        The process in this module is purification > super-resolution > frcnn
    
    Returns:
        Rendered HTML template with form and media URL if available.
    """
    form = CreatePost()
    media_url = None

    if request.method == "POST" and form.validate_on_submit():
        file = request.files.get("uploadImg")
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_IMG_FOLDER, filename)
        annot_path = os.path.join(ANNOT_IMG_FOLDER, f"pred_{filename}")
        file.save(upload_path)

        ext = os.path.splitext(filename)[1].lower()

        if ext in [".jpg", ".jpeg", ".png"]:
            pil = Image.open(upload_path).convert("RGB")

            print("enter pur")
            purified = run_purification(pil)
            print("enter esrgan")
            enhanced = run_realesrgan(purified)
            print("enter drcnn")
            result_img, class_names = detect_image(enhanced) 
            print("exity frcnn")
            result_img.save(annot_path)
            media_url = url_for("static", filename=f"annotated/pred_{filename}")
            print("Detected:", class_names)

            if class_names in COCO_CLASSES.values():
                if class_names in ["breast", "anus", "female_genital", "male_genital", "sexual_content", "harmful_object", "self_harm", "toxic_substance", "violence"]:
                    print("NSFW content detected")
                    if os.path.exists(upload_path):
                        os.remove(upload_path)
                    print(class_names)

        elif ext in [".mp4", ".avi", ".mov"]:
            # TODO: similar frame-wise pipeline (purify -> resshift -> detection)
            # TODO: capable si esrgan on video resolution
            result_vid, class_names = detect_video(upload_path, annot_path)
            media_url = url_for("static", filename=f"annotated/pred_{filename}")
            print("Detected:", class_names)

    return render_template("main.html", form=form, media_url=media_url)
