import os

from flask import Blueprint, render_template, url_for, session, redirect
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename

bp = Blueprint("routes", __name__)

# Initialize folders
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
       uploaded_file = form.uploadImg.data
       filename = secure_filename(uploaded_file.filename)
       save_path = os.path.join(UPLOAD_FOLDER, filename)
       uploaded_file.save(save_path)
       
       session["index"] = {"uploadImg": filename}
       return redirect(url_for("routes.result", filename=filename))
    return render_template("index.html", form=form)

# Route: Result
@bp.route("/result/<filename>")
def result(filename):
    return render_template("result.html", filename=filename)
