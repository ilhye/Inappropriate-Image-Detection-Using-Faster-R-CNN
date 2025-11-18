# 🔍 A Region-Based Convolutional Neural Network Approach to Detecting Harmful Cloaked Content for Automated Content Moderation
This project aims to detect inappropriate content in images using a pre-trained Faster R-CNN model. It leverages deep learning techniques and object detection pipelines to identify explicit or sensitive elements within an image frame.

## 🔍 Features
- Uses Faster R-CNN for object detection
- Filter inappropriate classes
- Image annotation and visualization for results
- Purify adversarial example

## 🧠 Models
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - image enhancement
- [Faster R-CNN](https://arxiv.org/abs/1506.01497) - main detection model
- [ResNet50](https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.config?fbclid=IwY2xjawK42MRleHRuA2FlbQIxMQABHlxnvp0hbHGu3sVw1fxfU8CEt6Vi89VKTLk8g-PFRmYqrDruKtNJjuTRby6B_aem_D9Z88sdQUl9KzVXh50sWpA) - backbone architecture

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
4. Setup modal
    ```bash
    python -m modal setup
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
> **Note**: This requires ffmpeg to be installed in your local machine. 
> You can get it here [FFmpeg Download](https://ffmpeg.org/download.html#build-windows)

## 🙌 Acknowledgements
We acknowledge the original development of Real-ESRGAN by [Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan](https://github.com/xinntao/Real-ESRGAN). We also recognize the partial implementation and contributions provided by [Igor Pavlov, Alex Wortoga, and Emily](https://github.com/ai-forever/Real-ESRGAN).