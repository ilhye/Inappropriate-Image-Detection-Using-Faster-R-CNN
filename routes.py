import os

from flask import Blueprint, render_template, redirect, session, url_for, flash, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, StringField, PasswordField, BooleanField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from PIL import Image
from frcnn import draw_boxes, contains_label

bp = Blueprint("routes", __name__)

# Initialize folders
UPLOAD_FOLDER = "uploads"
ANNOT_FOLDER = os.path.join("static", "annotated")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOT_FOLDER, exist_ok=True)

# Create Post
class CreatePost(FlaskForm):
    uploadImg = FileField(
        "Upload File",
        validators=[FileRequired("Please enter file"), FileAllowed(
            ["jpg", "jpeg", "png"], "Images only!")],
    )
    submit = SubmitField("Submit")

@bp.route("/", methods=["GET", "POST"])
def content_moderation():
    form = CreatePost()
    image_url = None

    print("entrypoint")
    if request.method == "POST" and form.validate_on_submit():
        print("entered")
        file = request.files.get("uploadImg")
        pil_img = Image.open(file.stream).convert("RGB")

        # Has violence == not save to upload
        # if contains_label(pil_img, "violence", score_thresh=0.8):
        #     flash("Upload rejected: Image contains violence")
        #     return redirect(url_for("routes.content_moderation"))

        result_img = draw_boxes(pil_img.copy())

        # No violence == annotate and save
        filename = secure_filename(file.filename)
        save_path = os.path.join(ANNOT_FOLDER, f"annot_{filename}")
        result_img.save(save_path)

        # return redirect(url_for("routes.result", filename=f"annot_{filename}"))
        image_url = url_for("static", filename=f"annotated/annot_{filename}")
    
    print("exit")
    return render_template("main.html", form=form, image_url=image_url)