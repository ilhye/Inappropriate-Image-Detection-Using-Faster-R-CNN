import os
import numpy as np
import torch
import cv2

from flask import Blueprint, render_template, request, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from utils import convert_to_np  
from cocoClass import COCO_CLASSES
from frcnn import detect_image, detect_video
# from realesrgan_wrapper import load_model as esrgan_load_model, run_sr as esrgan_run_sr
from purify.purification import Purifier
from purify.realesrgan import RealESRGANWrapper
# from purify.test_purify import AdversarialPatchPurifier

bp = Blueprint("routes", __name__)

# -----------------------------
# Folders
# -----------------------------
UPLOAD_IMG_FOLDER = os.path.join("static", "uploads")
ANNOT_IMG_FOLDER = os.path.join("static", "annotated")
os.makedirs(UPLOAD_IMG_FOLDER, exist_ok=True)
os.makedirs(ANNOT_IMG_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# _esrgan_model = esrgan_load_model(device=DEVICE)

# -----------------------------
# Flask form
# -----------------------------
class CreatePost(FlaskForm):
    uploadImg = FileField(
        "Upload File",
        validators=[
            FileRequired(),
            FileAllowed(["jpg", "jpeg", "png", "mp4", "avi", "mov"], "Images/Videos only"),
        ],
    )
    reset = SubmitField("Reset")
    submit = SubmitField("Submit")

# -----------------------------
# Super-resolution
# -----------------------------
# def run_realesrgan(image_pil: Image.Image) -> Image.Image:
#     return esrgan_run_sr(_esrgan_model, image_pil)

# -----------------------------
# Main Route
# -----------------------------
@bp.route("/", methods=["GET", "POST"])
def content_moderation():
    form = CreatePost()
    media_url = None
    message = ""
    # read button text from either query string or POST form data
    # request.values merges args and form so it covers both GET and POST
    text = request.values.get('button_text')
    print(f"Button text: {text}")

    if request.method == "POST" and form.validate_on_submit():
        file = request.files.get("uploadImg")
        filename = secure_filename(file.filename)

        # Save images/videos and still media when passed to pipeline
        upload_path = os.path.join(UPLOAD_IMG_FOLDER, filename)
        annot_path = os.path.join(ANNOT_IMG_FOLDER, f"pred_{filename}")
        file.save(upload_path)
        print(f"File type: {type(upload_path)}")

        ext = os.path.splitext(filename)[1].lower()

        # ---------- IMAGE PROCESSING----------
        if ext in [".jpg", ".jpeg", ".png"]:
            pil = Image.open(upload_path).convert("RGB")
            print(f"File type: {type(pil)}")

            print("Starting advanced purification...")
            processed_pil = Purifier.process(pil)

            print("Starting super-resolution...")
            enhanced = RealESRGANWrapper.enhance(processed_pil)
            # enhanced = np.array(enhanced)
            # h, w = enhanced.shape[:2]
            # enhanced_resized = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_CUBIC)
            # enhanced_resized = cv2.cvtColor(enhanced_resized, cv2.COLOR_BGR2RGB)

            print("Starting object detection...")
            result_img, class_names = detect_image(enhanced) 
            print("Detection complete")

            result_img.save(annot_path)
            media_url = url_for("static", filename=f"annotated/pred_{filename}")
            print("Detected:", class_names)

            if any(name in COCO_CLASSES.values() for name in class_names):
                print("NSFW content detected:", class_names)
                message = f"Contains NSFW content: {', '.join(class_names)}\n"
                os.remove(upload_path)
            else:
                message = "Safe content detected."

        # ---------- VIDEO PROCESSING ----------
        elif ext in [".mp4", ".avi", ".mov"]:
            result_vid, class_names = detect_video(upload_path, annot_path)
            media_url = url_for("static", filename=f"annotated/pred_{filename}")

            if any(name in COCO_CLASSES.values() for name in class_names):
                print("NSFW content detected:", class_names)
                message = f"Contains NSFW content: {', '.join(class_names)}\n"
                os.remove(upload_path)
            else:
                message = "Safe content detected."
                
            print("Detected classes:", class_names)

    return render_template("main.html", form=form, media_url=media_url, message=message)
