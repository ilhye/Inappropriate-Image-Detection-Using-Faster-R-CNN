"""
===========================================================
Program: Main
Programmer/s: Cristina C. Villasor
Date Written: June 15, 2025
Last Revised: Nov. 14, 2025

Purpose: Entry point for the Flask application

Program Fits in the General System Design:
- Entry point of the system
- Registers routes for handling URL
===========================================================
"""
from flask import Flask
from routes import bp as routes_bp
from dotenv import load_dotenv

import modal
import os

load_dotenv()  # Load .env file

app = modal.App("coin")  # Initialize Modal app

uploads_volume = modal.Volume.from_name("uploads-storage", create_if_missing=True)
annot_volume = modal.Volume.from_name("annot-storage", create_if_missing=True)

# Create dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "flask",
        "flask-wtf",
        "wtforms",
        "torch",
        "torchvision",
        "werkzeug",
        "huggingface-hub",
        "transformers",
        "Pillow",
        "opencv-contrib-python-headless",
        "tensorflow",
        "numpy",
        "scipy",
        "python-dotenv",
        "modal",
    )
    .add_local_dir(
        local_path=".",
        remote_path="/root",
        ignore=[".git", "__pycache__", "venv", "node_modules/",
                "static/uploads", "static/annotated", ".gitignore"]
    )
)

flask_app = Flask(__name__)                           # Create Flask app
flask_app.config["SECRET_KEY"] = os.getenv("CONFIG")  # Set config key for CSRF and Modal
flask_app.config["UPLOADS_VOLUME"] = uploads_volume
flask_app.config["ANNOT_VOLUME"] = annot_volume
flask_app.register_blueprint(routes_bp)               # Register routes blueprint

# Define modal function
@app.function(image=image, gpu="T4", timeout=1800, volumes={"/root/static/uploads": uploads_volume,
                                                            "/root/static/annotated": annot_volume})
@modal.wsgi_app()                                     # Create WSGI webpoint
def modal_app():                                      # Serve Flask app on Modal
    """Serve Flask app on Modal"""
    return flask_app

if __name__ == "__main__":              # Run Flask app locally
    flask_app.run(debug=True)