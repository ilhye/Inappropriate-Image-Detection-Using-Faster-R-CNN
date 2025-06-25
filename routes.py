import os

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from PIL import Image
from frcnn import draw_boxes, contains_label

bp = Blueprint("routes", __name__)

# Initialize folders
UPLOAD_FOLDER = "uploads"
ANNOT_FOLDER = os.path.join("static", "annotated")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOT_FOLDER, exist_ok=True)

# Form
class CreatePost(FlaskForm):
    uploadImg = FileField(
        "Upload Image",
        validators=[FileRequired(), FileAllowed(["jpg", "jpeg", "png"], "Images only!")],
    )
    submit = SubmitField("Post")

# Route
@bp.route("/", methods=["GET", "POST"])
def index():
    form = CreatePost()
    if form.validate_on_submit():
        file = request.files.get("uploadImg")
        pil_img = Image.open(file.stream).convert("RGB")

        # Has person == not save to upload
        if contains_label(pil_img, "violence", score_thresh=0.8):
            flash("Upload rejected: Image contains violence")
            return redirect(url_for("routes.index"))
        
        result_img = draw_boxes(pil_img.copy())

        # No person == annotate and save
        filename = secure_filename(file.filename)
        save_path = os.path.join(ANNOT_FOLDER, f"annot_{filename}")
        result_img.save(save_path)

        return redirect(url_for("routes.result", filename=f"annot_{filename}"))

    return render_template("index.html", form=form)

# Route: Result
@bp.route("/result/<path:filename>")
def result(filename):
    image_url = url_for("static", filename=f"annotated/{filename}")
    return render_template("result.html", image_url=image_url)