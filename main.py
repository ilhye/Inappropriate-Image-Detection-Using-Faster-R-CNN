"""
===========================================================
Program: Main
Programmer/s: Cristina C. Villasor
Date Written: June 15, 2025
Last Revised: Oct. 21, 2025

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

load_dotenv() # Load .env file

app = modal.App("inco-flask-app") # Initialize Modal app

# Create dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision==0.23.0")
    .pip_install(
        "flask",
        "flask-wtf",
        "wtforms",
        "werkzeug",
        "huggingface-hub",
        "transformers",
        "Pillow",
        "opencv-contrib-python-headless",
        "tensorflow",
        "numpy",
        "scipy",
        "python-dotenv",
        "tqdm",
        "blobfile",
        "xformers",
        "accelerate",
        "modal",
        "scikit-image"
    )
    .add_local_dir(
        local_path=".",
        remote_path="/root",
        ignore=[".git", "__pycache__", "venv", "static/uploads/", "static/annotated/", "node_modules/"]
    )
)

flask_app = Flask(__name__)                          # Create Flask app
flask_app.config["SECRET_KEY"] = os.getenv("CONFIG") # Set config key
flask_app.secret_key = os.getenv("SECRET_KEY")       # Set secret key
flask_app.register_blueprint(routes_bp)              # Register routes blueprint

@app.function(image=image, gpu="T4")    # Define modal function 
@modal.wsgi_app()                       # Create WSGI webpoint
def modal_app():                        # Serve Flask app on Modal
    """Serve Flask app on Modal"""
    return flask_app

if __name__ == "__main__":              # Run Flask app locally
    flask_app.run(debug=True)
