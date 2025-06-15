import os

from flask import Blueprint, render_template, redirect, url_for, flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from PIL import Image
from frcnn import contains_label, annotate

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
        file_storage = form.uploadImg.data
        pil_img = Image.open(file_storage.stream).convert("RGB")

        # Has person == not save to upload
        if contains_label(pil_img, "person"):
            flash("Upload rejected: a person was detected in the image.")
            return redirect(url_for("routes.index"))

        # No person == annotate and save
        file_storage.stream.seek(0)
        orig_fname = secure_filename(file_storage.filename)
        annotated_name = f"annot_{orig_fname}"
        save_path = os.path.join(ANNOT_FOLDER, annotated_name)

        annotated_img = annotate(pil_img.copy())
        annotated_img.save(save_path)

        return redirect(url_for("routes.result", filename=annotated_name))

    return render_template("index.html", form=form)

# Route: Result
@bp.route("/result/<path:filename>")
def result(filename):
    image_url = url_for("static", filename=f"annotated/{filename}")
    return render_template("result.html", image_url=image_url)