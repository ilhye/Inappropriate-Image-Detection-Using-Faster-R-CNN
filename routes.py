"""
===========================================================
Program: Routes
Programmer/s: Cristina C. Villasor
Date Written: June 15, 2025
Last Revised: Oct. 21, 2025

Purpose: Handles image and video uploads and maps a specific URL for the frontend. 

Program Fits in the General System Design:
- Defines the starting point of the image and video process
- Provide connection between the backend and frontend

Algorithm: 
- Reads input, if image, performs purification, super-resolution, and object detection. 
- If video, process by every 10th frame, then purification, super-resolution, and object detection. 
- Each detection will undergo VQA, then a final score will be created.
- If score is less than 0.8, then media is safe.
- Else the media will be filtered. 

Data Structures and Controls: 
- Uses a list to avoid repeating classes
- Uses if-else condition to process correctly the image/video
===========================================================
"""
import os
import torch

from flask import Blueprint, render_template, request, url_for, current_app
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from PIL import Image
from frcnn import detect_image, detect_video
from purify.purification import Purifier
from purify.realesrgan import RealESRGANWrapper

bp = Blueprint("routes", __name__)  # Blueprint for routes

# UPLOAD_IMG_FOLDER = os.path.join("static", "uploads")   # Local folders for uploaded images/videos
# ANNOT_IMG_FOLDER = os.path.join("static", "annotated")  # Local folder for annotated images/videos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_IMG_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ANNOT_IMG_FOLDER = os.path.join(BASE_DIR, "static", "annotated")
os.makedirs(UPLOAD_IMG_FOLDER, exist_ok=True)
os.makedirs(ANNOT_IMG_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Input form for uploading video and image
class CreatePost(FlaskForm):
    uploadImg = FileField(
        "Upload File",
        validators=[
            FileRequired(),
            FileAllowed(["jpg", "jpeg", "png", "mp4", "avi",
                        "mov", "mp4v"], "Images/Videos only"),
        ],
    )
    reset = SubmitField("Reset")
    submit = SubmitField("Submit")

# Main route - content moderation


@bp.route("/", methods=["GET", "POST"])
def content_moderation():
    form = CreatePost()      # Input form instance
    output_url = None         # For displaying annotated media
    message = ""             # For displaying safe/inappropriate message
    score_threshold = 0.8    # Final threshold after VQA and object detection

    text = request.values.get('button_text')
    print(f"Button text: {text}")

    if request.method == "POST" and form.validate_on_submit():
        file = request.files.get("uploadImg")
        filename = secure_filename(file.filename)

        # Save images/video to uploads folder
        upload_path = os.path.join(UPLOAD_IMG_FOLDER, filename)
        annot_path = os.path.join(ANNOT_IMG_FOLDER, f"pred_{filename}")
        file.save(upload_path)
        print(f"File type: {type(upload_path)}")

        # Get file extension
        ext = os.path.splitext(filename)[1].lower()

        # Image processing: purification -> super-resolution -> object detection
        if ext in [".jpg", ".jpeg", ".png"]:
            pil = Image.open(upload_path).convert("RGB")            # Load image
            processed_pil = Purifier.process(pil)                   # Purification
            enhanced = RealESRGANWrapper.enhance(processed_pil)     # Super-resolution
            result_img, class_names, score = detect_image(enhanced) # Object detection
            result_img.save(annot_path)                             # Save annotated image

            output_url = url_for(
                "static", filename=f"annotated/pred_{filename}")
            print("Detected:", class_names)

            # Filter image based on the score
            if score_threshold < score:
                os.remove(upload_path)
                message = f"Contains Inappropriate content: {','.join(list(dict.fromkeys(class_names)))}\nSuggested Actions: Content Removal or User Warning"
            else:
                message = "Content appears to be safe"

        # Video processing: get frames -> purifiacation -> super-resolution -> object detection -> repeat
        elif ext in [".mp4", ".avi", ".mov", ".mp4v"]:
            result_vid, class_names, scores = detect_video(upload_path, annot_path) # Object detection
            output_url = url_for(
                "static", filename=f"annotated/pred_{filename}")

            # Filter video based on the score
            if score_threshold < scores:
                os.remove(upload_path)
                message = f"Contains Inappropriate content: {','.join(list(dict.fromkeys(class_names)))}\nSuggested Actions: Content Removal or User Warning"
            else:
                message = "Content appears to be safe"

    return render_template("main.html", form=form, output_url=output_url, message=message)
