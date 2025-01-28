import pytest
import numpy as np
import cv2 as cv

from src.frame_processor import FrameProcessor  # Adjust import based on your project structure

@pytest.fixture
def frame_processor():
    """
    Pytest fixture to create a FrameProcessor instance 
    that can be reused across tests.
    """
    return FrameProcessor(target_width=640, target_height=640)

def test_preprocess_frame_valid_input(frame_processor):
    """
    Test that a valid frame is processed correctly.
    """
    # Create a dummy frame with shape (480, 640, 3) and random content.
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    preprocessed = frame_processor.preprocess_frame(dummy_frame)

    # Expected shape is (1, target_height, target_width, 3)
    assert preprocessed.shape == (1, 640, 640, 3), \
        f"Expected shape (1, 640, 640, 3), got {preprocessed.shape}"

    # Ensure values are in range [0, 1]
    assert preprocessed.min() >= 0.0 and preprocessed.max() <= 1.0, \
        "Preprocessed frame values are not in [0, 1] range."

def test_preprocess_frame_invalid_input_none(frame_processor):
    """
    Test that providing None as a frame raises ValueError.
    """
    with pytest.raises(ValueError):
        frame_processor.preprocess_frame(None)

def test_preprocess_frame_invalid_input_empty(frame_processor):
    """
    Test that providing an empty (zero-size) frame raises ValueError.
    """
    empty_frame = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError):
        frame_processor.preprocess_frame(empty_frame)

def test_preprocess_frames_batch(frame_processor):
    """
    Test preprocessing multiple frames in a batch.
    """
    # Create two dummy frames of different original sizes for testing.
    dummy_frame_1 = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    dummy_frame_2 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    frames = [dummy_frame_1, dummy_frame_2]
    preprocessed_batch = frame_processor.preprocess_frames(frames)

    # The batch size should be 2, and each frame is resized to 640x640 with 3 channels.
    assert preprocessed_batch.shape == (2, 640, 640, 3), \
        f"Expected shape (2, 640, 640, 3), got {preprocessed_batch.shape}"

    # Ensure values are in the [0, 1] range.
    assert preprocessed_batch.min() >= 0.0 and preprocessed_batch.max() <= 1.0, \
        "Preprocessed batch frame values are not in [0, 1] range."

def test_preprocess_frames_invalid_list(frame_processor):
    """
    Test that providing an invalid frames list raises ValueError.
    """
    # Not a list
    not_a_list = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        frame_processor.preprocess_frames(not_a_list)

    # Empty list
    with pytest.raises(ValueError):
        frame_processor.preprocess_frames([])

