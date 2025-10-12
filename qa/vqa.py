import io
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
from qa.questions import questions
from cocoClass import COCO_CLASSES

#Loading the model and tokenizer
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# can identify art but author will be unknown to it
# can't identify a violent scene
# sometimes it answers yes to other questions even if it's a no 
# (In this case, modify the question from question.py)
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

        for qa in questions:
            encoding = processor(image, qa, return_tensors="pt")

            # Forward pass
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = model.config.id2label[idx]

            answers.append(answer)

        return answers
    except Exception as e:
        # Return stringified error for easy debugging in callers
        return str(e)

def decision(classes, answer, detection_score):
    """ Make decision based on the answers and detected classes
    Args: 
        classes (list): List of detected class names from Faster R-CNN
        answer (list): List of answers from the VQA model
        detection_score (float): Highest confidence score from Faster R-CNN detections

        if image is art, it will pass the filtering if more than 2 harmful indicators are present
    Returns:
        total_score (float): Combined score indicating likelihood of inappropriate content
    """
    is_art = answer[0].lower() == "yes" if answer else False
    print("Answer", answer)
    if is_art: # Handles art context
        harmful_indicators = sum(1 for ans in answer[1:10] if ans.lower() == "yes")
        print("Harmful:", harmful_indicators)
        if harmful_indicators >= 2 and any(name in COCO_CLASSES.values() for name in classes):
            return 1.0 # Flag as inappropriate if more than 2 COCO classes detected
        elif harmful_indicators == 1:
            return 0.5 # Flag as mildly inappropriate if 1 COCO class detected
        else:
             return 0.0 # Safe
    else: # Handles non-art context
        harmful_indicators = sum(1 for ans in answer[1:10] if ans.lower() == "yes") # Count yes answers from q3 to q12
        harmful_score = min(1.0, 0.5 + (harmful_indicators * 0.1))

        # Faster R-CNN score and if relevant COCO classes found, boost by 0.2
        coco_boost = 0.2 if any(name in COCO_CLASSES.values() for name in classes) else 0.0
        detection_component = min(detection_score + coco_boost, 1.0)

    # Weighted average ensemble
    total_score = (
        (0.3 * harmful_score) +       
        (0.7 * detection_component)
    )
    return float(total_score)