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
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

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
            encoding = processor(image, qa, return_tensors="pt")

            # Forward pass 
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = model.config.id2label[idx]
            confidence = logits.softmax(-1).max().item()
            print(f"Q: {qa} A: {answer} (confidence: {confidence:.4f})")
            answers.append(answer)
            confidences.append(confidence)

        return answers, confidences
    except Exception as e:
        # Return stringified error for easy debugging in callers
        return str(e)

def decision(classes, answers, vqa_confidences, detection_score):
    """ Make decision based on the answers and detected classes
    Args: 
        classes (list): List of detected class names from Faster R-CNN
        answers (list): List of answers from the VQA model
        vqa_confidences (list): List of confidence scores from the VQA model
        detection_score (float): Highest confidence score from Faster R-CNN detections

        if image is art, it will pass the filtering if more than 2 harmful indicators are present
    Returns:
        total_score (float): Combined score indicating likelihood of inappropriate content
    """
    
    # Calculate harmful score based on yes answers
    harmful_confs = [conf for ans, conf in zip(answers[2:12], vqa_confidences[2:12]) if ans.lower() == "yes"]
    harmful_score = np.mean(harmful_confs) if harmful_confs else 0.0
    print("Harmful confidence list:", harmful_confs, "→ Harmful score:", harmful_score)

    # Identify art and educational context
    is_art = answers[0].lower() == "yes" and vqa_confidences[0] > 0.8
    is_educational = answers[1].lower() == "yes" and vqa_confidences[1] > 0.8

    # Adjust harmful score based on context
    if is_art:
        harmful_score *= 0.5 # 50% reduction for art
    if is_educational:
        harmful_score *= 0.5 # 50% reduction for educational

    # Boost faster r-cnn score if relevant coco classes found
    coco_boost = 0.15 if any(name in COCO_CLASSES.values() for name in classes) else 0.0
    detection_component = min(detection_score + coco_boost, 1.0)
    print("Detection component:", detection_component)

    # Weighted averaged based on VQA and Faster R-CNN confidence
    vqa_weight = 0.3 + (np.mean(vqa_confidences) * 0.2)  # between 0.3–0.5
    det_weight = 1.0 - vqa_weight
    print(f"Weights => VQA: {vqa_weight:.2f}, Detection: {det_weight:.2f}")

    # Weighted average ensemble
    total_score = (vqa_weight * harmful_score) + (det_weight * detection_component)
    # total_score = float(np.clip(total_score, 0.0, 1.0)) # For T4
    # total_score = float(torch.clamp(total_score, 0.0, 1.0).item()) # For B200

    if isinstance(total_score, torch.Tensor):
        # move to CPU, ensure float
        total_score = float(torch.clamp(total_score, 0.0, 1.0).item())
    else:
        total_score = float(np.clip(total_score, 0.0, 1.0))
        
    print("Final total score:", total_score)

    return total_score
