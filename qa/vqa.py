"""
===========================================================
Program: VQA
Programmer/s: Catherine Joy R. Pailden and Cristina C. Villasor
Date Written: Oct 11, 2025
Last Revised: Nov. 13, 2025

Purpose: Verify if the image contains inappropriate content and create the final score.

Program Fits in the General System Design:
- This program is used after object detection
- The answers from VQA and detected classes from object detecetion are used to create a final score
- The score produced here is used by 'Routes' to filter inappropriate content

Algorithm: 
- Takes the image, list of questions, and detected classes as input
- Loads the pre-trained VQA model and tokenizer
- Calculate average confidence score for questions with 'yes' answers
- If the image is an art or educational, reduce confidence score
- If image contains non-empty classes, increase confidence score. 
- Then, calculate the total confidence score with 50% trust for VQA and object detection. 

Data Structures and Controls: 
- Uses if-else condition to switch condition
===========================================================
"""
import io
import numpy as np
import os
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
from qa.questions import questions
from cocoClass import COCO_CLASSES

#Loading the model and tokenizer
PROCESSOR = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
MODEL = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_answer(image, question=questions):
    """ Get answer based on the questions
    Args:
        image (PIL.Image): Input image in PIL format
        question (list): List of questions to ask the model
    Returns:
        answers (list): List of answers from the model
    """
    try:
        # img = _to_pil(image)
        print("Type vqa:", type(image))
        answers = []
        confidences = []

        for qa in questions:
            encoding = PROCESSOR(image, qa, return_tensors="pt") # Prepare inputs

            outputs = MODEL(**encoding)                   # Get model outputs
            logits = outputs.logits                       # Extract logits
            idx = logits.argmax(-1).item()                # Get index of highest logit
            answer = MODEL.config.id2label[idx]           # Get answers
            confidence = logits.softmax(-1).max().item()  # Get confidence score

            print(f"Q: {qa} A: {answer} (confidence: {confidence:.4f})")
            answers.append(answer)
            confidences.append(confidence)

        return answers, confidences
    except Exception as e:
        return str(e)

def decision(classes, answers, vqa_confidences, detection_score):
    """ Make decision based on the answers and detected classes
    Args: 
        classes (list): List of detected class names from Faster R-CNN
        answers (list): List of answers from the VQA model
        vqa_confidences (list): List of confidence scores from the VQA model
        detection_score (float): Highest confidence score from Faster R-CNN detections
    Returns:
        total_score (float): Combined score indicating likelihood of inappropriate content
    """
    # Calculate harmful score and boosts detection
    harmful_avg = vqa_conf(answers, vqa_confidences)
    detection_component = obj_detection_conf(detection_score, classes)

    # Weighted averaged based on VQA and Faster R-CNN confidence
    vqa_weight = 0.3 + (np.mean(vqa_confidences) * 0.2)  # between 0.3–0.5
    det_weight = 1.0 - vqa_weight
    print(f"Weights => VQA: {vqa_weight:.2f}, Detection: {det_weight:.2f}")

    # Weighted average ensemble
    total_score = (vqa_weight * harmful_avg) + (det_weight * detection_component)

    if isinstance(total_score, torch.Tensor):
        # Move to CPU, ensure float
        total_score = float(torch.clamp(total_score, 0.0, 1.0).item())
    else:
        total_score = float(np.clip(total_score, 0.0, 1.0))
        
    print("Total score:", total_score)

    return total_score

def vqa_conf(answers, vqa_confidences):
    """
    Calculate harmful score based on yes answers
    Args:
        answers (list): List of answers from the VQA model
        vqa_confidences (list): List of confidence scores from the VQA model
    Returns:
        harmful_avg (float): Average confidence score for harmful content
    """
    harmful_confs = [conf for ans, conf in zip(answers[2:12], vqa_confidences[2:12]) if ans.lower() == "yes"]
    harmful_avg = np.mean(harmful_confs) if harmful_confs else 0.0
    print("Harmful score:", harmful_avg)

    # Identify art and educational context
    is_art = answers[0].lower() == "yes" and vqa_confidences[0] > 0.8
    is_educational = answers[1].lower() == "yes" and vqa_confidences[1] > 0.8
    
    if is_art:
        harmful_avg *= 0.5 # 50% reduction for art
    if is_educational:
        harmful_avg *= 0.5 # 50% reduction for educational

    return harmful_avg

def obj_detection_conf(detection_score, classes):
    """
    Calculate detection component with COCO class boost
    Args:
        detection_score (float): Highest confidence score from Faster R-CNN detections
        classes (list): List of detected class names from Faster R-CNN
    Returns:
        detection_component (float/tensor): Adjusted detection confidence score
    """
    coco_boost = 0.15 if any(name in COCO_CLASSES.values() for name in classes) else 0.0
    detection_component = min(detection_score + coco_boost, 1.0)
    print("Detection component:", detection_component)
    return detection_component