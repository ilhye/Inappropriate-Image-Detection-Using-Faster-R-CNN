import os

from flask import Blueprint, render_template, redirect, session, url_for, flash, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, StringField, PasswordField, BooleanField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from PIL import Image
from frcnn import detect_image, detect_video

bp = Blueprint("routes", __name__)

# Initialize folders
UPLOAD_IMG_FOLDER = os.path.join("static", "uploads")
ANNOT_IMG_FOLDER = os.path.join("static", "annotated")
os.makedirs(UPLOAD_IMG_FOLDER, exist_ok=True)
os.makedirs(ANNOT_IMG_FOLDER, exist_ok=True)

# Create Post


class CreatePost(FlaskForm):
    uploadImg = FileField(
        "Upload File",
        validators=[FileRequired("Please enter file"), FileAllowed(
            ["jpg", "jpeg", "png", "mp4", "mkv", "avi", "mov"], "Images and Videos only")],
    )
    submit = SubmitField("Submit")


def file_type(file):
    file_name, file_extension = os.path.splitext(file.filename)
    print(f"Extension:{file_extension}")
    return file_extension[1:].lower()


@bp.route("/", methods=["GET", "POST"])
def content_moderation():
    form = CreatePost()
    media_url = None

    print("entrypoint")
    if request.method == "POST" and form.validate_on_submit():
        print("entered")

        file = request.files.get("uploadImg")
        filename = secure_filename(file.filename)
        annot_path = os.path.join(ANNOT_IMG_FOLDER, f"pred_{filename}")
        upload_path = os.path.join(UPLOAD_IMG_FOLDER, f"safe_{filename}")
        file.save(upload_path)

        file_ext = file_type(file)

        if file_ext in ["jpg", "jpeg", "png"]:
            result_img, class_names = detect_image(file)
            result_img.save(annot_path)
            media_url = url_for("static", filename=f"img-annotated/pred_{filename}")
            
            print("image received")
            print("Detected classes:", class_names)

            # TODO: Change class names based on LSPD and VHD11K
            # TODO: LSPD - breast, anus, female_genital, male_genital
            # TODO: VHD11K - harmful_object, toxic_substance, violence, suicide, safe, sexual_content
            if "non_violence" in class_names:
                print("Non-Violence content detected")

            if "violence" in class_names:
                print("Violence content detected")
                if os.path.exists(upload_path):
                    os.remove(upload_path)
                print(class_names)

        elif file_ext in ["mp4", "mkv", "avi", "mov"]:         
            result_vid, class_names = detect_video(upload_path, annot_path)
            result_vid.save(annot_path)
            media_url = url_for("static", filename=f"vid-annotated/pred_{filename}")

            print("video received")
            print("Detected classes:", class_names)

            # TODO: Change class names based on LSPD and VHD11K
            # TODO: LSPD - breast, anus, female_genital, male_genital
            # TODO: VHD11K - harmful_object, toxic_substance, violence, suicide, safe, sexual_content
            if "non_violence" in class_names:
                print("Non-Violence content detected")
            
            if "violence" in class_names:
                print("Violence content detected")
                if os.path.exists(upload_path):
                    os.remove(upload_path)
    print("exit")
    return render_template("main.html", form=form, media_url=media_url)