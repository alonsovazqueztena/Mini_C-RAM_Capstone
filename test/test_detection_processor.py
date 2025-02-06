# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 3, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a19036-b408-800e-8419-7a79dd969fcc
# (used as starter code for basic functionality).

# We import the Python testing library.
import pytest

# We import the DetectionProcessor class from the source code.
from src.detection_processor import DetectionProcessor

# We are testing the processing of detections 
# with default parameters.
def test_process_detections_default():
    """Testing DetectionProcessor with default parameters: 
    confidence_threshold = 0.5 and target_classes = []."""

    # We have three detections with different boundary boxes, 
    # confidences, and classes.
    detections = [
        {"bbox": [10, 20, 30, 40], "confidence": 0.6, "class_id": 1},
        {"bbox": [15, 25, 35, 45], "confidence": 0.4, "class_id": 2},
        {"bbox": [5, 5, 15, 15], "confidence": 0.9, "class_id": 2},
    ]
    
    # We initialize the DetectionProcessor.
    processor = DetectionProcessor()

    # We process the detections.
    processed = processor.process_detections(
        detections)
    
    # We expect the first and third detection only 
    # (confidences: 0.6, 0.9 >= 0.5).
    assert len(processed) == 2

    # We check the first detection of its
    # boundary box, confidence, and class ID.
    assert processed[0]["bbox"] == [10, 20, 30, 40]
    assert processed[0]["confidence"] == 0.6
    assert processed[0]["class_id"] == 1

    # We check the first detection's centroid: 
    # ((10+30)/2, (20+40)/2) = (20, 30).
    assert processed[0]["centroid"] == (20, 30)

    # We check the second detection of its
    # boundary box, confidence, and class ID.
    assert processed[1]["bbox"] == [5, 5, 15, 15]
    assert processed[1]["confidence"] == 0.9
    assert processed[1]["class_id"] == 2

    # We check the second detection's centroid: 
    # ((5+15)/2, (5+15)/2) = (10, 10)
    assert processed[1]["centroid"] == (10, 10)

# We are testing the processing of detections 
# with target classes.
def test_process_detections_with_target_classes():
    """Testing the DetectionProcessor when specific 
    target_classes are provided."""

    # We have three detections with different boundary boxes, 
    # confidences, and classes.
    detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.8, "class_id": 0},
        {"bbox": [10, 20, 30, 40], "confidence": 0.9, "class_id": 1},
        {"bbox": [20, 30, 40, 50], "confidence": 0.95, "class_id": 2},
    ]

    # We set the target classes to only include class_id = 1.
    target_classes = [
        1]

    # We initialize the DetectionProcessor with the target classes 
    # and the default confidence threshold.
    processor = DetectionProcessor(
        target_classes=target_classes, 
        confidence_threshold=0.5)
    
    # We process the detections.
    processed = processor.process_detections(
        detections)

    # Only the second detection should remain due to its class_id.
    assert len(processed) == 1

    # We check the second detection's class ID, confidence, 
    # and boundary box.
    assert processed[0]["class_id"] == 1
    assert processed[0]["confidence"] == 0.9
    assert processed[0]["bbox"] == [10, 20, 30, 40]

    # We check the second detection's centroid:
    assert processed[0]["centroid"] == ((10 + 30) / 2, (20 + 40) / 2)

# We are testing the processing of detections with a 
# higher confidence threshold.
def test_process_detections_custom_confidence_threshold():
    """Testing DetectionProcessor with a higher confidence threshold."""

    # We have three detections with different boundary boxes,
    # confidences, and classes.
    detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.5, "class_id": 0},
        {"bbox": [10, 20, 30, 40], "confidence": 0.7, "class_id": 1},
        {"bbox": [20, 30, 40, 50], "confidence": 0.9, "class_id": 2},
    ]

    # We set the confidence threshold to 0.8.
    processor = DetectionProcessor(
        confidence_threshold=0.8)
    
    # We process the detections.
    processed = processor.process_detections(
        detections)

    # Only the third detection (confidence=0.9) will pass.
    assert len(processed) == 1

    # We check the third detection's class ID, 
    # confidence, and boundary box.
    assert processed[0]["confidence"] == 0.9
    assert processed[0]["class_id"] == 2
    assert processed[0]["bbox"] == [20, 30, 40, 50]

    # We check the third detection's centroid:
    assert processed[0]["centroid"] == ((20 + 40) / 2, (30 + 50) / 2)

# We are testing the processing of an empty list of detections.
def test_process_detections_empty_input():
    """Testing DetectionProcessor with an empty list of detections."""

    # We initialize the DetectionProcessor.
    processor = DetectionProcessor()

    # We process an empty list of detections.
    processed = processor.process_detections(
        [])
    
    # We expect an empty list to be returned.
    assert processed == []

# We are testing the processing of detections
# where no detections met the filter.
def test_process_detections_no_detections_after_filter():
    """Testing DetectionProcessor where no detections survive filtering."""

    # We have two detections with confidence below the threshold.
    detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.3, "class_id": 0},
        {"bbox": [10, 20, 30, 40], "confidence": 0.2, "class_id": 1}
    ]

    # We initialize the DetectionProcessor with a high confidence threshold.
    processor = DetectionProcessor(
        confidence_threshold=0.5)
    
    # We process the detections.
    processed = processor.process_detections(
        detections)
    
    # No detections should pass the filter.
    assert len(processed) == 0