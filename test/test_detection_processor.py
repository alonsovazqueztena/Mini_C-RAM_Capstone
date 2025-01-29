import pytest
from src.detection_processor import DetectionProcessor

def test_process_detections_default():
    """
    Test DetectionProcessor with default parameters:
    - confidence_threshold = 0.5
    - target_classes = []
    We expect only detections with confidence >= 0.5 to be returned, regardless of class_id.
    """
    detections = [
        {"bbox": [10, 20, 30, 40], "confidence": 0.6, "class_id": 1},
        {"bbox": [15, 25, 35, 45], "confidence": 0.4, "class_id": 2},
        {"bbox": [5, 5, 15, 15], "confidence": 0.9, "class_id": 2},
    ]
    
    processor = DetectionProcessor()
    processed = processor.process_detections(detections)
    
    # We expect the first and third detection only (confidences: 0.6, 0.9 >= 0.5)
    assert len(processed) == 2

    # Check first detection
    assert processed[0]["bbox"] == [10, 20, 30, 40]
    assert processed[0]["confidence"] == 0.6
    assert processed[0]["class_id"] == 1
    # Centroid: ((10+30)/2, (20+40)/2) = (20, 30)
    assert processed[0]["centroid"] == (20, 30)

    # Check second detection
    assert processed[1]["bbox"] == [5, 5, 15, 15]
    assert processed[1]["confidence"] == 0.9
    assert processed[1]["class_id"] == 2
    # Centroid: ((5+15)/2, (5+15)/2) = (10, 10)
    assert processed[1]["centroid"] == (10, 10)


def test_process_detections_with_target_classes():
    """
    Test DetectionProcessor when specific target_classes are provided.
    """
    detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.8, "class_id": 0},
        {"bbox": [10, 20, 30, 40], "confidence": 0.9, "class_id": 1},
        {"bbox": [20, 30, 40, 50], "confidence": 0.95, "class_id": 2},
    ]
    target_classes = [1]  # Only keep class_id = 1

    processor = DetectionProcessor(target_classes=target_classes, confidence_threshold=0.5)
    processed = processor.process_detections(detections)

    # Only the second detection should remain, because only class_id=1 is targeted
    assert len(processed) == 1
    assert processed[0]["class_id"] == 1
    assert processed[0]["confidence"] == 0.9
    assert processed[0]["bbox"] == [10, 20, 30, 40]
    # Centroid check
    assert processed[0]["centroid"] == ((10 + 30) / 2, (20 + 40) / 2)


def test_process_detections_custom_confidence_threshold():
    """
    Test DetectionProcessor with a higher confidence threshold.
    """
    detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.5, "class_id": 0},
        {"bbox": [10, 20, 30, 40], "confidence": 0.7, "class_id": 1},
        {"bbox": [20, 30, 40, 50], "confidence": 0.9, "class_id": 2},
    ]

    # Set threshold to 0.8
    processor = DetectionProcessor(confidence_threshold=0.8)
    processed = processor.process_detections(detections)

    # We expect only the third detection (confidence=0.9) to pass
    assert len(processed) == 1
    assert processed[0]["confidence"] == 0.9
    assert processed[0]["class_id"] == 2


def test_process_detections_empty_input():
    """
    Test DetectionProcessor with an empty list of detections.
    Should return an empty list.
    """
    processor = DetectionProcessor()
    processed = processor.process_detections([])
    assert processed == []


def test_process_detections_no_detections_after_filter():
    """
    Test DetectionProcessor where no detections survive filtering.
    In this case, we'll make confidence too low for all detections.
    """
    detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.3, "class_id": 0},
        {"bbox": [10, 20, 30, 40], "confidence": 0.2, "class_id": 1}
    ]

    processor = DetectionProcessor(confidence_threshold=0.5)
    processed = processor.process_detections(detections)
    assert len(processed) == 0
