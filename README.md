# Inappropriate Image Detection Using Faster R-CNN

This project aims to detect inappropriate content in images using a pre-trained Faster R-CNN model. It leverages deep learning techniques and object detection pipelines to identify explicit or sensitive elements within an image frame.

## Features

- Uses Faster R-CNN for object detection
- Filter inappropriate classes
- Image annotation and visualization

## Models
- [ResNet50](https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.config?fbclid=IwY2xjawK42MRleHRuA2FlbQIxMQABHlxnvp0hbHGu3sVw1fxfU8CEt6Vi89VKTLk8g-PFRmYqrDruKtNJjuTRby6B_aem_D9Z88sdQUl9KzVXh50sWpA)
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)

## Datasets
- [VHD11K](https://arxiv.org/abs/2409.19734)
- [LSPD](https://sites.google.com/uit.edu.vn/LSPD)


## Installation

1. **Create a virtual environment**
   ```bash
   python -m venv venv
2. Activate the virtual environment
    ```bash
    venv\Scripts\activate
3. Install required dependencies
    ```bash
    pip install -r requirements.txt
4. Run the application
    ```bash
    python main.py

## Credits
Inspired by [snapscope](https://github.com/ErolGelbul/snapscope), developed by Erol Gelbul. We acknowledge its influence in the development of our approach and thank the author for the foundational work.
