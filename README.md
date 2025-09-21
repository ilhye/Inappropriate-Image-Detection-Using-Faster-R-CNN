# 🚫 Inappropriate Image Detection Using Faster R-CNN

This project aims to detect inappropriate content in images using a pre-trained Faster R-CNN model. It leverages deep learning techniques and object detection pipelines to identify explicit or sensitive elements within an image frame.

## ✨ Features

- Faster R-CNN–based object detection
- Filtering of inappropriate classes (e.g., explicit or violent content)
- Image annotation & visualization for detected objects

## 🧠 Models
- [ResNet50](https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.config?fbclid=IwY2xjawK42MRleHRuA2FlbQIxMQABHlxnvp0hbHGu3sVw1fxfU8CEt6Vi89VKTLk8g-PFRmYqrDruKtNJjuTRby6B_aem_D9Z88sdQUl9KzVXh50sWpA)
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [ResShift]()
- [Adversarial Purification]()

## 📦 Requirements
* Python
* pip
* Node.js and npm
* Taiwind CSS

## 📂 Datasets
- [VHD11K]()
- [LSPD]()

## ⚙️ Installation

1. Clone the repository
    ```bash
    git clone https://github.com/ilhye/Inappropriate-Image-Detection-Using-Faster-R-CNN.git
1. **Create a virtual environment**
   ```bash
   python -m venv venv
2. Activate the virtual environment
    ```bash
    venv\Scripts\activate
3. Install Python dependencies
    ```bash
    pip install -r requirements.txt
4. Set up Tailwind
    ```bash
    npm init -y
    npm install tailwindcss @tailwind/cli
5. Compile Tailwind CSS
    ```bash
    npx @tailwindcss/cli -i ./static/css/input.css -o ./static/css/output.css --watch
4. Run the application
    ```bash
    python main.py
## 🙌 Credits
Inspired by [snapscope](https://github.com/ErolGelbul/snapscope), developed by Erol Gelbul. We acknowledge its influence in the development of our approach and thank the author for the foundational work.
