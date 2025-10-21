# 🔍 A Region-Based Convolutional Neural Network Approach to Detecting Harmful Cloaked Content for Automated Content Moderation
This project aims to detect inappropriate content in images using a pre-trained Faster R-CNN model. It leverages deep learning techniques and object detection pipelines to identify explicit or sensitive elements within an image frame.

## 🔍 Features
- Uses Faster R-CNN for object detection
- Filter inappropriate classes
- Image annotation and visualization for results

## 🧠 Models
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - image enhancement
- [ResNet50](https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.config?fbclid=IwY2xjawK42MRleHRuA2FlbQIxMQABHlxnvp0hbHGu3sVw1fxfU8CEt6Vi89VKTLk8g-PFRmYqrDruKtNJjuTRby6B_aem_D9Z88sdQUl9KzVXh50sWpA) - backbone architecture
- [Faster R-CNN](https://arxiv.org/abs/1506.01497) - main detection model

## 🗂️ Datasets
To maintain class balance, 10K images were selected from each dataset (20K total).
Using RoboFlow, we applied data augmentation (horizontal/vertical flips), expanding the dataset to 40K images.
- [Large Scale Porngraphic Dataset](https://sites.google.com/uit.edu.vn/LSPD): 50K annotated images
- [Harmful Object Detection Dataset](https://github.com/poori-nuna/HOD-Benchmark-Dataset): 10k images

## ⚙️ Installation
1. Create a virtual environment
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment
   ```bash
   venv\Scripts\activate
   ```
3. Install required dependencies
    - Local (with GPU)
        ```bash
        pip install -r requirements.txt
         ```
    - Using Modal (serverless)
        ```bash 
        modal serve main.py
        ```
4. Install node modules
   ```bash
   npm install
   ```
5. Run the application
    - Local
        ```bash
        python main.py
        ```
    - Modal
        ```bash 
        modal serve main.py
        ```

## 🙌 Credits
Inspired by [snapscope](https://github.com/ErolGelbul/snapscope), developed by Erol Gelbul. We acknowledge its influence in the development of our approach and thank the author for the foundational work.
