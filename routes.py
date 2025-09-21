import os

from flask import Blueprint, render_template, redirect, session, url_for, flash, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, StringField, PasswordField, BooleanField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from PIL import Image
from frcnn import detect_image, detect_video

# New imports for purification/resshift pipeline
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
try:
    # import the local purification wrapper 
    from purification import adversarial_anti_aliasing, adversarial_purification, DiffusionModel
except Exception:
    adversarial_anti_aliasing = None
    adversarial_purification = None
    DiffusionModel = None

# Import ResShift inference helpers
_resshift_module = None
_resshift_predict_fn = None
try:
    import ResShift.predict as _rs_pred
    _resshift_module = _rs_pred
except Exception:
    try:
        import ResShift.inference_resshift as _rs_inf
        _resshift_module = _rs_inf
    except Exception:
        _resshift_module = None

if _resshift_module is not None:
    # find a plausible entrypoint function
    for name in ("predict", "inference", "run", "main", "enhance"):
        if hasattr(_resshift_module, name):
            _resshift_predict_fn = getattr(_resshift_module, name)
            break

# optional cv2 for video frame processing 
try:
    import cv2
except Exception:
    cv2 = None

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

# Reuse a Diffusion Model instance to avoid repeated init costs
_diffusion_model = None
def get_diffusion_model():
    global _diffusion_model
    if _diffusion_model is None and DiffusionModel is not None:
        # try to locate a checkpoint in the GuidedDiffusionPur folder
        default_ckpt = os.path.join(os.path.dirname(__file__), "..", "GuidedDiffusionPur", "models", "256x256_diffusion.pt")
        default_ckpt = os.path.abspath(default_ckpt)
        if not os.path.exists(default_ckpt):
            default_ckpt = None
        try:
            _diffusion_model = DiffusionModel(model_path=default_ckpt, image_size=256)
        except Exception:
            # fallback to placeholder model
            _diffusion_model = DiffusionModel(model_path=None, image_size=256)
    return _diffusion_model

def apply_resshift(pil_image):
    """
    Try to apply ResShift enhancement on a PIL image using any discovered entrypoint.
    If no ResShift implementation is available or it fails, return the original image.
    The ResShift function signatures in this repo may vary; this helper attempts to call
    the function with common single-image signatures (PIL or numpy array).
    """
    if _resshift_predict_fn is None:
        return pil_image

    try:
        # try PIL first
        out = _resshift_predict_fn(pil_image)
        if isinstance(out, Image.Image):
            return out
        # try numpy array
        if isinstance(out, np.ndarray):
            return Image.fromarray(out)
        # some implementations return path string
        if isinstance(out, str) and os.path.exists(out):
            return Image.open(out).convert("RGB")
    except Exception:
        # try passing numpy array
        try:
            arr = np.array(pil_image)
            out = _resshift_predict_fn(arr)
            if isinstance(out, Image.Image):
                return out
            if isinstance(out, np.ndarray):
                return Image.fromarray(out)
        except Exception:
            pass

    # fallback
    return pil_image

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

        # prepare diffusion model
        diffusion_model = get_diffusion_model()

        if file_ext in ["jpg", "jpeg", "png"]:
            try:
                # Load saved image -> tensor -> anti-alias -> purify
                pil = Image.open(upload_path).convert("RGB")
                img_t = ToTensor()(pil).unsqueeze(0)  # [1,C,H,W], values [0,1]

                # anti-alias 
                aa = adversarial_anti_aliasing(img_t, sigma=1.0) if adversarial_anti_aliasing is not None else img_t

                # purification with guided diffusion 
                purified_t = adversarial_purification(aa, diffusion_model) if adversarial_purification is not None and diffusion_model is not None else aa

                # convert to PIL
                pil_purified = ToPILImage()(purified_t.squeeze(0))

                # enhancement via ResShift 
                pil_enhanced = apply_resshift(pil_purified)

                # save enhanced purified image and pass to detection
                purified_path = os.path.join(UPLOAD_IMG_FOLDER, f"purified_{filename}")
                pil_enhanced.save(purified_path)

                result_img, class_names = detect_image(purified_path)
                result_img.save(annot_path)
                media_url = url_for("static", filename=f"annotated/pred_{filename}")

                print("image received (purified -> resshift -> frcnn)")
                print("Detected classes:", class_names)

                if "non_violence" in class_names:
                    print("Non-Violence content detected")

                if "violence" in class_names:
                    print("Violence content detected")
                    if os.path.exists(upload_path):
                        os.remove(upload_path)
                    print(class_names)
            except Exception as ex:
                print("Image purification/resshift/detection failed:", ex)
                try:
                    result_img, class_names = detect_image(upload_path)
                    result_img.save(annot_path)
                    media_url = url_for("static", filename=f"annotated/pred_{filename}")
                except Exception as ex2:
                    print("Fallback detect_image also failed:", ex2)

        elif file_ext in ["mp4", "mkv", "avi", "mov"]:
            try:
                # If cv2 available and purification pipeline available, process frame-wise
                if cv2 is not None and adversarial_anti_aliasing is not None and adversarial_purification is not None and diffusion_model is not None:
                    cap = cv2.VideoCapture(upload_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    purified_video_path = os.path.join(UPLOAD_IMG_FOLDER, f"purified_{filename}")
                    out = cv2.VideoWriter(purified_video_path, fourcc, fps, (width, height))

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # BGR -> RGB -> PIL -> tensor
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(frame_rgb)
                        t = ToTensor()(pil_frame).unsqueeze(0)

                        aa = adversarial_anti_aliasing(t, sigma=1.0) if adversarial_anti_aliasing else t
                        purified_t = adversarial_purification(aa, diffusion_model) if adversarial_purification else aa

                        pil_p = ToPILImage()(purified_t.squeeze(0)).resize((width, height))
                        # enhance frame with ResShift 
                        pil_p = apply_resshift(pil_p)

                        frame_out = cv2.cvtColor(np.array(pil_p), cv2.COLOR_RGB2BGR)
                        out.write(frame_out)

                    cap.release()
                    out.release()

                    # pass purified video path to detection
                    result_vid, class_names = detect_video(purified_video_path, annot_path)
                    if os.path.exists(annot_path):
                        media_url = url_for("static", filename=f"annotated/pred_{filename}")
                    else:
                        media_url = url_for("static", filename=f"uploads/purified_{filename}")

                    print("video received (frame-wise purified -> resshift -> frcnn)")
                    print("Detected classes:", class_names)
                    if "violence" in class_names:
                        if os.path.exists(upload_path):
                            os.remove(upload_path)
                else:
                    # fallback: call detect_video on original upload
                    result_vid, class_names = detect_video(upload_path, annot_path)
                    try:
                        result_vid.save(annot_path)
                    except Exception:
                        pass
                    media_url = url_for("static", filename=f"annotated/pred_{filename}")

                    print("video received (fallback)")
                    print("Detected classes:", class_names)

                    if "non_violence" in class_names:
                        print("Non-Violence content detected")

                    if "violence" in class_names:
                        print("Violence content detected")
                        if os.path.exists(upload_path):
                            os.remove(upload_path)
            except Exception as ex:
                print("Video purification/detection failed:", ex)
                # fallback to original behavior
                try:
                    result_vid, class_names = detect_video(upload_path, annot_path)
                    try:
                        result_vid.save(annot_path)
                    except Exception:
                        pass
                    media_url = url_for("static", filename=f"annotated/pred_{filename}")
                except Exception as ex2:
                    print("Fallback detect_video also failed:", ex2)
    print("exit")
    return render_template("main.html", form=form, media_url=media_url)