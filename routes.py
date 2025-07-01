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
        validators=[FileRequired("Please enter file"), FileAllowed(["jpg", "jpeg", "png"], "Images only!")],
    )
    submit = SubmitField("Post")

# Login Form
class Login(FlaskForm):
    username = StringField('Username', validators=[
                           DataRequired("Please enter your username")])
    password = PasswordField('Password', validators=[
                             DataRequired("Please enter your password")])
    submit = SubmitField('Sign In')

class NewAccount(FlaskForm):
    username = StringField('Username', validators=[
                           DataRequired("Please enter your username")])
    password = PasswordField('Password', validators=[
                             DataRequired("Please enter your password")])
    submit = SubmitField('Sign In')

# Route
@bp.route("/post", methods=["GET", "POST"])
def index():
    username = session.get("username")
    form = CreatePost()
    if form.validate_on_submit():
        file = request.files.get("uploadImg")
        pil_img = Image.open(file.stream).convert("RGB")

        # Has violence == not save to upload
        if contains_label(pil_img, "violence", score_thresh=0.8):
            flash("Upload rejected: Image contains violence")
            return redirect(url_for("routes.index"))
        
        result_img = draw_boxes(pil_img.copy())

        # No violence == annotate and save
        filename = secure_filename(file.filename)
        save_path = os.path.join(ANNOT_FOLDER, f"annot_{filename}")
        result_img.save(save_path)

        return redirect(url_for("routes.result", filename=f"annot_{filename}", username=username))

    return render_template("index.html", form=form, username=username)

# Route: Result
@bp.route("/result/<path:filename>")
def result(filename):
    image_url = url_for("static", filename=f"annotated/{filename}")
    return render_template("result.html", image_url=image_url)

@bp.route("/", methods=["GET", "POST"])
def login():
    form = Login()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        session["username"] = username
        return redirect(url_for("routes.index"))
    return render_template("signIn.html", form=form)

@bp.route("/signUp", methods=["GET", "POST"])
def signUp():
    form = NewAccount()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        session["username"] = username
        return redirect(url_for("routes.index"))
    return render_template("signUp.html", form=form)