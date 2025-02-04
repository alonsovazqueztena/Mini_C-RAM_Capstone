# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 2, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/6797eb56-3dd8-800e-946e-816dcd9e5c0e
# (used as starter code for basic functionality).

import pytest
import logging
import cv2 as cv
import numpy as np
from unittest.mock import MagicMock, patch

# Import the class to be tested
from src.video_stream_manager import VideoStreamManager

@pytest.fixture
def manager():
    """Fixture to create a fresh instance of VideoStreamManager for each test."""
    return VideoStreamManager(capture_device=0, frame_width=848, frame_height=480)

def test_init(manager):
    """Test that the VideoStreamManager is initialized correctly."""
    assert manager.capture_device == 0
    assert manager.frame_width == 848
    assert manager.frame_height == 480
    assert manager.capture is None  # Should not be opened yet.

@patch.object(cv, 'VideoCapture')
def test_initialize_stream_success(mock_VideoCapture, manager):
    """
    Test that initialize_stream() opens the capture device successfully
    and sets the width/height/properties.
    """
    mock_capture = MagicMock()
    # Mock the return value of cv2.VideoCapture(...) to be our MagicMock capture
    mock_VideoCapture.return_value = mock_capture

    # Make isOpened() return True, simulating a successful open
    mock_capture.isOpened.return_value = True

    manager.initialize_stream()

    # Check that cv2.VideoCapture() was called with the correct device index
    mock_VideoCapture.assert_called_once_with(0)

    # Check that .set(...) was called with correct width, height, hardware accel
    mock_capture.set.assert_any_call(cv.CAP_PROP_FRAME_WIDTH, 848)
    mock_capture.set.assert_any_call(cv.CAP_PROP_FRAME_HEIGHT, 480)
    mock_capture.set.assert_any_call(cv.CAP_PROP_HW_ACCELERATION, cv.VIDEO_ACCELERATION_ANY)

    # Assert that it stored the capture object in manager.capture
    assert manager.capture == mock_capture

@patch.object(cv, 'VideoCapture')
def test_initialize_stream_failure(mock_VideoCapture, manager):
    """
    Test that initialize_stream() raises an error if the capture device
    cannot be opened.
    """
    mock_capture = MagicMock()
    mock_VideoCapture.return_value = mock_capture

    # Make isOpened() return False, simulating a failed open
    mock_capture.isOpened.return_value = False

    with pytest.raises(RuntimeError, match="Cannot open the HDMI capture card"):
        manager.initialize_stream()

@patch.object(cv, 'VideoCapture')
def test_get_frame_success(mock_VideoCapture, manager):
    """
    Test that get_frame() returns the captured frame when everything works.
    """
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    # Return a dummy numpy array so we can call frame.shape
    fake_frame = np.zeros((480, 848, 3), dtype=np.uint8)
    mock_capture.read.return_value = (True, fake_frame)
    mock_VideoCapture.return_value = mock_capture

    manager.initialize_stream()

    frame = manager.get_frame()
    assert frame is not None
    assert frame.shape == (480, 848, 3)

@patch.object(cv, 'VideoCapture')
def test_get_frame_no_stream(mock_VideoCapture, manager):
    """
    Test that get_frame() raises an error if the stream is not initialized.
    """
    # Notice we do *not* call manager.initialize_stream() first.
    # So the capture is never actually opened.
    with pytest.raises(RuntimeError, match="The video stream cannot be initialized"):
        manager.get_frame()

@patch.object(cv, 'VideoCapture')
def test_get_frame_read_failure(mock_VideoCapture, manager):
    """
    Test that get_frame() returns None (and logs an error) if .read() fails.
    """
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.read.return_value = (False, None)  # ret=False
    mock_VideoCapture.return_value = mock_capture

    manager.initialize_stream()

    frame = manager.get_frame()
    assert frame is None

@patch.object(cv, 'VideoCapture')
def test_get_frame_invalid_frame(mock_VideoCapture, manager):
    """
    Test that get_frame() returns None if the captured frame is None.
    """
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.read.return_value = (True, None)  # ret=True but frame=None
    mock_VideoCapture.return_value = mock_capture

    manager.initialize_stream()

    frame = manager.get_frame()
    assert frame is None

@patch.object(cv, 'VideoCapture')
def test_release_stream(mock_VideoCapture, manager):
    """
    Test that release_stream() properly calls .release() on the capture object
    when the stream is open.
    """
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_VideoCapture.return_value = mock_capture

    manager.initialize_stream()
    manager.release_stream()

    # Check that .release() was called exactly once
    mock_capture.release.assert_called_once()

def test_release_stream_no_capture(manager):
    """
    Test that release_stream() does not raise an error
    if there is no open capture.
    """
    # manager.capture is None because we never opened it
    # This should gracefully do nothing.
    manager.release_stream()

def test_context_manager(manager):
    """
    Test using the `with` statement to ensure __enter__ and __exit__ are called.
    We'll mock initialize_stream and release_stream calls to verify.
    """
    with patch.object(manager, 'initialize_stream') as mock_init, \
         patch.object(manager, 'release_stream') as mock_release:
        with manager as m:
            # Inside the 'with', initialize_stream should have been called
            mock_init.assert_called_once()
            # The returned object should be the manager itself
            assert m is manager

        # After the 'with' block, release_stream should be called
        mock_release.assert_called_once()
