# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 3, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a18fe4-56ec-800e-bf00-4af40519d328
# (used as starter code for basic functionality).

# # We import the Python testing library.
import pytest

# We import the NumPy library to work with mock frames.
import numpy as np

# We import the FrameProcessor class from the source code.
from src.frame_processor import FrameProcessor 

# We are creating a Pytest fixture to 
# create a FrameProcessor instance.
@pytest.fixture
def frame_processor():
    """Creating a fixture to create a FrameProcessor instance 
    that can be reused."""

    # We are creating a FrameProcessor instance with 
    # a target width and heigh of 640.
    return FrameProcessor(
        target_width=640, 
        target_height=640)

# We are testing the preprocessing of a valid frame.
def test_preprocess_frame_valid_input(
        frame_processor):
    """Testing that a valid frame is processed correctly."""

    # This creates a dummy frame with shape (480, 640, 3).
    dummy_frame = np.random.randint(
        0, 256, (480, 640, 3), 
        dtype=np.uint8)

    # We preprocess the frame.
    preprocessed = frame_processor.preprocess_frame(dummy_frame)

    # We expect the frame shape to have the following: 
    # (1 batch dimension, target_height, target_width, 
    # 3 channels for RGB).
    assert preprocessed.shape == (1, 640, 640, 3), \
        f"Expected shape (1, 640, 640, 3), got {preprocessed.shape}"

    # We ensure that the pixel values are normalized.
    assert preprocessed.min() >= 0.0 and preprocessed.max() <= 1.0, \
        "Preprocessed frame values are not in [0, 1] range."

# We are testing the preprocessing of an invalid frame.
def test_preprocess_frame_invalid_input_none(
        frame_processor):
    """Testing that providing None as a frame raises ValueError."""

    # We expect a ValueError to be raised.
    with pytest.raises(
            ValueError):
        frame_processor.preprocess_frame(
            None)

# We are testing the preprocessing of an empty frame.
def test_preprocess_frame_invalid_input_empty(
        frame_processor):
    """Testing that providing an empty frame raises ValueError."""

    # We create an empty frame.
    empty_frame = np.array(
        [], dtype=np.uint8)
    
    # We expect a ValueError to be raised.
    with pytest.raises(
            ValueError):
        frame_processor.preprocess_frame(
            empty_frame)

# We are testing the preprocessing of a batch of frames.
def test_preprocess_frames_batch(
        frame_processor):
    """Testing the preprocessing of multiple frames in a batch."""

    # This creates two dummy frames of 
    # different original sizes for testing.
    dummy_frame_1 = np.random.randint(
        0, 256, (720, 1280, 3), 
        dtype=np.uint8)
    dummy_frame_2 = np.random.randint(
        0, 256, (480, 640, 3), 
        dtype=np.uint8)

    # We preprocess the frames.
    frames = [
        dummy_frame_1, 
        dummy_frame_2]
    preprocessed_batch = frame_processor.preprocess_frames(
        frames)

    # The batch size should be 2, and each frame 
    # is resized to 640x640 with 3 channels.
    assert preprocessed_batch.shape == (2, 640, 640, 3), \
        f"Expected shape (2, 640, 640, 3), got {preprocessed_batch.shape}"

    # This ensures pixel values are normalized.
    assert preprocessed_batch.min() >= 0.0 and preprocessed_batch.max() <= 1.0, \
        "Preprocessed batch frame values are not in [0, 1] range."

# We are testing the preprocessing of an invalid list of frames.
def test_preprocess_frames_invalid_list(
        frame_processor):
    """Testing that providing an invalid frames 
    list raises ValueError."""

    # We create an invalid list.
    not_a_list = np.zeros(
        (480, 640, 3), 
        dtype=np.uint8)
    
    # We expect a ValueError to be raised for an invalid list.
    with pytest.raises(
            ValueError):
        frame_processor.preprocess_frames(
            not_a_list)

    # We expect a ValueError to be raised for an empty list.
    with pytest.raises(ValueError):
        frame_processor.preprocess_frames(
            [])