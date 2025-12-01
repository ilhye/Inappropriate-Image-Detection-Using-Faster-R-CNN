"""
===========================================================
Program: cocoClass.py
Programmer/s: Cristina C. Villasor
Date Written: June 15, 2025
Last Revised: Nov. 19, 2025

Purpose: Custom COCO classes for inappropriate content detection.

Program Fits in the General System Design:
- Use when frcnn.py starts the object detection process
- Provides class names for detected objects

Data Structures and Controls: 
- Uses a list for storing classes
===========================================================
"""
COCO_CLASSES = { 1: "alcohol",
    2: "anus",
    3: "blood",
    4: "breast",
    5: "cigarette",
    6: "female_genital",
    7: "gun",
    8: "insulting_gesture",
    9: "knife",
    10: "male_genital" }