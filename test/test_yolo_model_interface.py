# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 3, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a18fe4-56ec-800e-bf00-4af40519d328
# (used as starter code for basic functionality).

# We import the Python testing library.
import pytest

# We import NumPy to work with arrays in boundary boxes.
import numpy as np

# We can use the `MagicMock` class from the `unittest.mock` 
# module to create mock objects.
from unittest.mock import patch, MagicMock

# We import the YOLOModelInterface class from the source code.
from src.yolo_model_interface import YOLOModelInterface

# We are testing the initialization of the YOLO model.
@patch(
        "src.yolo_model_interface.YOLO")
def test_init_model_success(
    mock_yolo):
    """Testing that the model initializes successfully 
    when YOLO loads without error."""

    # We are creating a mock YOLO model.
    mock_yolo.return_value = MagicMock()

    # We are initializing the YOLO model interface.
    interface = YOLOModelInterface(
        model_path="fake_path.pt")
    
    # We expect the model to be assigned after a successful load.
    assert interface.model is not None, "Model should be assigned after successful load."

# We are testing the failed initialization of the YOLO model.
@patch(
        "src.yolo_model_interface.YOLO", 
        side_effect=Exception("Model load error"))
def test_init_model_fail(
    mock_yolo):
    """Testing that an exception is raised 
    when the YOLO model fails to load."""

    # We are expecting an exception to be 
    # raised if the model fails to load.
    with pytest.raises(Exception) as exc:
        YOLOModelInterface(model_path="fake_path.pt")

    # We expect the exception message to 
    # indicate that the model failed to load.
    assert "Model load error" in str(exc.value), \
        "Exception message should indicate the model failed to load."

# We are testing the predict method with no detections.
@patch(
        "src.yolo_model_interface.YOLO")
def test_predict_empty_result(
    mock_yolo):
    """Testing that predict returns an empty list 
    when the YOLO model returns no detections."""

    # This is the mock model.
    mock_model = MagicMock()

    # The model predicts no detections.
    mock_model.predict.return_value = []

    # We are mocking the YOLO model to return the mock model.
    mock_yolo.return_value = mock_model

    # We are initializing the YOLO model interface.
    interface = YOLOModelInterface(
        model_path="fake_path.pt")
    
    # A frame with no objects is created.
    frame = np.zeros(
        (640, 640, 3), 
        dtype=np.uint8)
    
    # We are running inference on the frame.
    detections = interface.predict(
        frame)

    # We expect an empty list to be returned.
    assert isinstance(detections, list), "Should return a list."

    # We expect no detections to be returned.
    assert len(detections) == 0, "No detections should be returned if predict returns an empty list."

# We are testing the predict method with valid detections.
@patch(
        "src.yolo_model_interface.YOLO")
def test_predict_with_results(
    mock_yolo):
    """Testing that predict parses and returns detections 
    correctly when YOLO returns valid results."""

    # This mocks a single detection box.
    mock_box = MagicMock()

    # The bounding box, confidence, and class ID are set.
    mock_box.xyxy = [
        np.array([10, 20, 30, 40], 
        dtype=np.float32)]
    mock_box.conf = np.array(
        [0.8], dtype=np.float32)
    mock_box.cls = np.array(
        [2], dtype=np.float32)

    # This mocks the result object containing the above box.
    mock_result = MagicMock()
    mock_result.boxes = [
        mock_box]

    # This mocks the model to predict and return one detection.
    mock_model = MagicMock()
    mock_model.predict.return_value = [
        mock_result]
    mock_yolo.return_value = mock_model

    # We are initializing the YOLO model interface.
    interface = YOLOModelInterface(
        model_path="fake_path.pt", 
        confidence_threshold=0.5)
    
    # We are creating a frame with the detected object.
    frame = np.zeros(
        (640, 640, 3), 
        dtype=np.uint8)
    
    # We are running inference on the frame.
    detections = interface.predict(
        frame)

    # We expect one detection to be returned.
    assert len(detections) == 1, "Should return exactly one detection."
    detection = detections[
        0]
    
    # We expect the detection to match the boundary box,
    # confidence, and class ID set above.
    assert detection["bbox"] == [10, 20, 30, 40], "Bounding box mismatch."
    assert detection["confidence"] == pytest.approx(0.8, abs=1e-5), "Confidence mismatch."
    assert detection["class_id"] == 2, "Class ID mismatch."

# We are testing the predict_batch method.
@patch(
        "src.yolo_model_interface.YOLO")
def test_predict_batch(mock_yolo):
    """Testing that predict_batch processes 
    multiple frames and returns the correct detections."""

    # This mocks first detection box.
    mock_box1 = MagicMock()

    # The bounding box, confidence, and class ID are set.
    mock_box1.xyxy = [
        np.array([10, 20, 30, 40], 
                 dtype=np.float32)]
    mock_box1.conf = np.array(
        [0.8], dtype=np.float32)
    mock_box1.cls = np.array(
        [2], dtype=np.float32)
    
    # This mocks the first result object containing the above box.
    mock_result1 = MagicMock()
    mock_result1.boxes = [
        mock_box1]

    # This mocks first detection box.
    mock_box2 = MagicMock()

    # The bounding box, confidence, and class ID are set.
    mock_box2.xyxy = [
        np.array([50, 60, 70, 80], 
                 dtype=np.float32)]
    mock_box2.conf = np.array(
        [0.9], dtype=np.float32)
    mock_box2.cls = np.array(
        [3], dtype=np.float32)
    
    # This mocks the second result object containing the above box.
    mock_result2 = MagicMock()
    mock_result2.boxes = [
        mock_box2]

    # This mocks a model to return two results (one for each frame).
    mock_model = MagicMock()
    mock_model.predict.return_value = [
        mock_result1, mock_result2]
    mock_yolo.return_value = mock_model

    # We are initializing the YOLO model interface.
    interface = YOLOModelInterface(
        model_path="fake_path.pt", 
        confidence_threshold=0.5)
    
    # We are creating two frames with detected objects.
    frames = [
        np.zeros((640, 640, 3), dtype=np.uint8),
        np.zeros((640, 640, 3), dtype=np.uint8)
    ]

    # We are running inference on the frames.
    all_detections = interface.predict_batch(frames)

    # We expect two frames of detections to be returned.
    assert len(all_detections) == 2, "Should return detections for both frames."

    # We hold the detections of the first frame.
    first_frame_detections = all_detections[
        0]

    # We expect one detection to be returned.
    assert len(first_frame_detections) == 1, "Should detect exactly one object in the first frame."

    # We expect the detection to match the boundary box, 
    # confidence, and class ID set above.
    assert first_frame_detections[0]["bbox"] == [10, 20, 30, 40]
    assert first_frame_detections[0]["confidence"] == pytest.approx(0.8, abs=1e-5), "Confidence mismatch."
    assert first_frame_detections[0]["class_id"] == 2

    # We hold the detections of the second frame.
    second_frame_detections = all_detections[
        1]
    
    # We expect one detection to be returned.
    assert len(second_frame_detections) == 1, "Should detect exactly one object in the second frame."

    # We expect the detection to match the boundary box, 
    # confidence, and class ID set above.
    assert second_frame_detections[0]["bbox"] == [50, 60, 70, 80]
    assert second_frame_detections[0]["confidence"] == pytest.approx(0.9, abs=1e-5), "Confidence mismatch."
    assert second_frame_detections[0]["class_id"] == 3