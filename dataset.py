import os

from roboflow import Roboflow
from dotenv import load_dotenv

rf = Roboflow(api_key=os.getenv("API_KEY"))
project = rf.workspace("securityviolence").project("violence-detection-p4qev")
version = project.version(4)
dataset = version.download("coco")