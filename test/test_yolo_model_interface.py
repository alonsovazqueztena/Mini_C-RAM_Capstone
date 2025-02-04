# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 2, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a18fe4-56ec-800e-bf00-4af40519d328
# (used as starter code for basic functionality).

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# If your class is in a file named 'yolo_model_interface.py', adjust the import accordingly:
from src.yolo_model_interface import YOLOModelInterface


@patch("src.yolo_model_interface.YOLO")
def test_init_model_success(mock_yolo):
    """
    Test that the model initializes successfully when YOLO loads without error.
    """
    mock_yolo.return_value = MagicMock()
    interface = YOLOModelInterface(model_path="fake_path.pt")
    assert interface.model is not None, "Model should be assigned after successful load."


@patch("src.yolo_model_interface.YOLO", side_effect=Exception("Model load error"))
def test_init_model_fail(mock_yolo):
    """
    Test that an exception is raised when the YOLO model fails to load.
    """
    with pytest.raises(Exception) as exc:
        YOLOModelInterface(model_path="fake_path.pt")
    assert "Model load error" in str(exc.value), \
        "Exception message should indicate the model failed to load."


@patch("src.yolo_model_interface.YOLO")
def test_predict_empty_result(mock_yolo):
    """
    Test that predict returns an empty list when the YOLO model returns no detections.
    """
    # Mock model and its predict method to return an empty list
    mock_model = MagicMock()
    mock_model.predict.return_value = []
    mock_yolo.return_value = mock_model

    interface = YOLOModelInterface(model_path="fake_path.pt")
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    detections = interface.predict(frame)
    assert isinstance(detections, list), "Should return a list."
    assert len(detections) == 0, "No detections should be returned if predict returns an empty list."


@patch("src.yolo_model_interface.YOLO")
def test_predict_with_results(mock_yolo):
    """
    Test that predict parses and returns detections correctly when YOLO returns valid results.
    """
    # Mocking a single detection box
    mock_box = MagicMock()
    mock_box.xyxy = [np.array([10, 20, 30, 40], dtype=np.float32)]  # bounding box
    mock_box.conf = np.array([0.8], dtype=np.float32)  # confidence
    mock_box.cls = np.array([2], dtype=np.float32)  # class ID

    # Mock result object containing the above box
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    # Mock the model so that predict returns one detection
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mock_yolo.return_value = mock_model

    interface = YOLOModelInterface(model_path="fake_path.pt", confidence_threshold=0.5)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    detections = interface.predict(frame)

    assert len(detections) == 1, "Should return exactly one detection."
    detection = detections[0]
    assert detection["bbox"] == [10, 20, 30, 40], "Bounding box mismatch."
    assert detection["confidence"] == pytest.approx(0.8, abs=1e-5), "Confidence mismatch."
    assert detection["class_id"] == 2, "Class ID mismatch."


@patch("src.yolo_model_interface.YOLO")
def test_predict_batch(mock_yolo):
    """
    Test that predict_batch processes multiple frames and returns the correct detections.
    """
    # Mocking first detection box
    mock_box1 = MagicMock()
    mock_box1.xyxy = [np.array([10, 20, 30, 40], dtype=np.float32)]
    mock_box1.conf = np.array([0.8], dtype=np.float32)
    mock_box1.cls = np.array([2], dtype=np.float32)
    mock_result1 = MagicMock()
    mock_result1.boxes = [mock_box1]

    # Mocking second detection box
    mock_box2 = MagicMock()
    mock_box2.xyxy = [np.array([50, 60, 70, 80], dtype=np.float32)]
    mock_box2.conf = np.array([0.9], dtype=np.float32)
    mock_box2.cls = np.array([3], dtype=np.float32)
    mock_result2 = MagicMock()
    mock_result2.boxes = [mock_box2]

    # Mock model to return two results (one for each frame)
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result1, mock_result2]
    mock_yolo.return_value = mock_model

    interface = YOLOModelInterface(model_path="fake_path.pt", confidence_threshold=0.5)
    frames = [
        np.zeros((640, 640, 3), dtype=np.uint8),
        np.zeros((640, 640, 3), dtype=np.uint8)
    ]

    all_detections = interface.predict_batch(frames)

    assert len(all_detections) == 2, "Should return detections for both frames."

    # Check detections for the first frame
    first_frame_detections = all_detections[0]
    assert len(first_frame_detections) == 1, "Should detect exactly one object in the first frame."
    assert first_frame_detections[0]["bbox"] == [10, 20, 30, 40]
    assert first_frame_detections[0]["confidence"] == pytest.approx(0.8, abs=1e-5), "Confidence mismatch."
    assert first_frame_detections[0]["class_id"] == 2

    # Check detections for the second frame
    second_frame_detections = all_detections[1]
    assert len(second_frame_detections) == 1, "Should detect exactly one object in the second frame."
    assert second_frame_detections[0]["bbox"] == [50, 60, 70, 80]
    assert second_frame_detections[0]["confidence"] == pytest.approx(0.9, abs=1e-5), "Confidence mismatch."
    assert second_frame_detections[0]["class_id"] == 3
