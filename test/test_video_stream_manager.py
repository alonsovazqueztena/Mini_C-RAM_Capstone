# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 3, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/6797eb56-3dd8-800e-946e-816dcd9e5c0e
# (used as starter code for basic functionality).

# We import the Python testing library.
import pytest

# Many of our tests rely on mocking objects (images, 
# the YOLO model, etc.).

# We can use the `MagicMock` class from the `unittest.mock` 
# module to create mock objects.
from unittest.mock import MagicMock, patch

# We use the OpenCV library to work with images.
import cv2 as cv

# We use this library to work with the frame shape.
import numpy as np

# We are imported the class to be tested.
from src.video_stream_manager import VideoStreamManager

# We are setting up a video stream manager 
# fixture for the unit tests.
@pytest.fixture
def manager():
    """Fixture to create a new instance of 
    VideoStreamManager for each test."""

    return VideoStreamManager(capture_device=0,
                              frame_width=848, frame_height=480
                              )

# We are testing the initialization of the VideoStreamManager class.
def test_init(
        manager):
    """Testing that the VideoStreamManager is initialized correctly."""

    # We are checking the default values of the VideoStreamManager.
    assert manager.capture_device == 0
    assert manager.frame_width == 848
    assert manager.frame_height == 480
    assert manager.capture is None

# Utilizing our fixture, we are testing 
# the initialization of the video stream.
@patch.object(
        cv, 'VideoCapture'
        )
def test_initialize_stream_success(
        mock_VideoCapture, manager):
    """Testing that initialize_stream() opens the 
    capture device successfully and sets the width/height/properties."""

    # We are creating a mock capture object.
    mock_capture = MagicMock()

    # We are mocking the return value to be our MagicMock capture.
    mock_VideoCapture.return_value = mock_capture

    # We make isOpened() return True to show a successful open.
    mock_capture.isOpened.return_value = True

    # We are initializing the video stream.
    manager.initialize_stream()

    # We check that cv2.VideoCapture() 
    # was called with the correct device index.
    mock_VideoCapture.assert_called_once_with(
        0
        )

    # Check that .set(...) was called with correct 
    # width, height, and hardware acceleration.
    mock_capture.set.assert_any_call(
        cv.CAP_PROP_FRAME_WIDTH, 848
        )
    mock_capture.set.assert_any_call(
        cv.CAP_PROP_FRAME_HEIGHT, 480
        )
    mock_capture.set.assert_any_call(
        cv.CAP_PROP_HW_ACCELERATION, 
        cv.VIDEO_ACCELERATION_ANY
        )

    # We assert that it stored the capture object in manager.capture.
    assert manager.capture == mock_capture

# We are testing the failed initialization of the video stream.
@patch.object(
        cv, 'VideoCapture'
        )
def test_initialize_stream_failure(
        mock_VideoCapture, manager):
    """Testing that initialize_stream() raises an error 
    if the capture device cannot be opened."""

    # We are creating a mock capture object.
    mock_capture = MagicMock()

    # We are mocking the return value to be our MagicMock capture.
    mock_VideoCapture.return_value = mock_capture

    # We make isOpened() return False to simulate a failed open.
    mock_capture.isOpened.return_value = False

    # We are testing that the HDMI capture card cannot be opened.
    with pytest.raises(
            RuntimeError, 
            match="Cannot open the HDMI capture card"):
        manager.initialize_stream()

# We are testing the successful capture of a frame.
@patch.object(
        cv, 'VideoCapture'
        )
def test_get_frame_success(
        mock_VideoCapture, manager):
    """Testing that get_frame() returns the 
    captured frame when everything works."""

    # We are creating a mock capture object.
    mock_capture = MagicMock()

    # We are mocking the capture object to be opened.
    mock_capture.isOpened.return_value = True

    # We return a dummy numpy array to call frame.shape.
    fake_frame = np.zeros(
        (480, 848, 3), 
        dtype=np.uint8)
    
    # We mock the read method to return True and the fake frame.
    mock_capture.read.return_value = (
        True, fake_frame)
    
    # We are mocking the return value to be our MagicMock capture.
    mock_VideoCapture.return_value = mock_capture

    # We are initializing the video stream.
    manager.initialize_stream()

    # We are getting the frame from the video stream.
    frame = manager.get_frame()

    # We are checking that the frame is not None.
    assert frame is not None

    # We are checking that the frame shape is correct.
    assert frame.shape == (
        480, 848, 3)

# We are testing the failed capture of a frame
# due to the stream not being initialized.
@patch.object(
        cv, 'VideoCapture')
def test_get_frame_no_stream(
        mock_VideoCapture, manager):
    """Testing that get_frame() raises an error 
    if the stream is not initialized."""
    
    # This is testing that the video stream is not initialized.
    with pytest.raises(
            RuntimeError, match="The video stream cannot be initialized"):
        manager.get_frame()

# We are testing the failed read of a frame.
@patch.object(
        cv, 'VideoCapture')
def test_get_frame_read_failure(
        mock_VideoCapture, manager):
    """Testing that get_frame() returns None if .read() fails."""

    # We are creating a mock capture object.
    mock_capture = MagicMock()

    # We are mocking the capture object to be opened.
    mock_capture.isOpened.return_value = True

    # We are mocking the read method to return False and None.
    mock_capture.read.return_value = (
        False, None)
    
    # We are mocking the return value to be our MagicMock capture.
    mock_VideoCapture.return_value = mock_capture

    # We are initializing the video stream.
    manager.initialize_stream()

    # We are getting the frame from the video stream.
    frame = manager.get_frame()

    # We are checking that the frame is None.
    assert frame is None

# We are testing the capture of an invalid frame.
@patch.object(
        cv, 'VideoCapture')
def test_get_frame_invalid_frame(
        mock_VideoCapture, manager):
    """Testing that get_frame() returns None 
    if the captured frame is None."""

    # We are creating a mock capture object.
    mock_capture = MagicMock()

    # We are mocking the capture object to be opened.
    mock_capture.isOpened.return_value = True

    # We are mocking the read method 
    # to return True but None for the frame.
    mock_capture.read.return_value = (
        True, None)

    # We are mocking the return value to be our MagicMock capture.
    mock_VideoCapture.return_value = mock_capture

    # We are initializing the video stream.
    manager.initialize_stream()

    # We are getting the frame from the video stream.
    frame = manager.get_frame()

    # We are checking that the frame is None.
    assert frame is None

# We are testing the release of the stream.
@patch.object(
        cv, 'VideoCapture')
def test_release_stream(
        mock_VideoCapture, manager):
    """Testing that release_stream() properly calls 
    .release() on the capture object when the stream is open."""

    # We are creating a mock capture object.
    mock_capture = MagicMock()

    # We are mocking the capture object to be opened.
    mock_capture.isOpened.return_value = True

    # We are mocking the return value to be our MagicMock capture.
    mock_VideoCapture.return_value = mock_capture

    # We are initializing the video stream.
    manager.initialize_stream()

    # We are releasing the video stream.
    manager.release_stream()

    # Check that .release() was called exactly once.
    mock_capture.release.assert_called_once()

# We are testing the release of the stream when there is no capture.
def test_release_stream_no_capture(
        manager):
    """Testing that release_stream() does not raise an error
    if there is no open capture."""

    # The manager.capture is None because we never opened it.
    # This does nothing.
    manager.release_stream()

# We are testing the context manager.
def test_context_manager(
        manager):
    """Testing using the `with` statement to 
    ensure __enter__ and __exit__ are called."""

    # We are mocking the initialize_stream and release_stream methods.
    with patch.object(manager, 'initialize_stream') as mock_init, \
         patch.object(manager, 'release_stream') as mock_release:
        
        # We are entering the context manager.
        with manager as m:
            
            # The initialize_stream method should be called.
            mock_init.assert_called_once()

            # The returned object should be the manager itself.
            assert m is manager

        # The release_stream should be called.
        mock_release.assert_called_once()