import os
import cv2 as cv
import logging
from video_stream_manager import VideoStreamManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure the Qt platform backend is set to 'xcb'
os.environ["QT_QPA_PLATFORM"] = "xcb"

def main():
    try:
        # Initialize the video stream manager with desired resolution
        manager = VideoStreamManager(capture_device=0, frame_width=640, frame_height=480)
        manager.initialize_stream()

        logging.info("Starting real-time video stream...")

        while True:
            frame = manager.get_frame()
            if frame is not None:
                # Display the frame in a window
                cv.imshow("Video Stream", frame)

                # Break the loop if 'q' is pressed
                if cv.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Exiting video stream...")
                    break
            else:
                logging.error("Captured an invalid frame.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        manager.release_stream()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
