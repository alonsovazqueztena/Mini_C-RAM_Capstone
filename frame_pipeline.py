import cv2 as cv
import numpy as np
import logging
from video_stream_manager import VideoStreamManager  # Ensure this class is accessible
from frame_processor import FrameProcessor  # Ensure this class is accessible

class FramePipeline:

    def __init__(self, capture_device=0, frame_width=1280, frame_height=720, target_width=640, target_height=640):
        self.video_stream = VideoStreamManager(capture_device, frame_width, frame_height)
        self.frame_processor = FrameProcessor(target_width, target_height)

    def run(self):
        try:
            # Initialize the video stream.
            with self.video_stream as stream:
                logging.info("Starting the frame processing pipeline.")

                while True:
                    # Capture a frame from the video stream.
                    frame = stream.get_frame()
                    if frame is None:
                        logging.warning("No frame captured. Exiting the pipeline.")
                        break

                    # Preprocess the frame using the FrameProcessor.
                    processed_frame = self.frame_processor.preprocess_frame(frame)

                    # Optional: Perform operations on the processed frame (e.g., YOLO model inference).
                    # For now, we'll just log the frame shape.
                    logging.info(f"Processed frame shape: {processed_frame.shape}")

                    # Optional: Display the original frame (for debugging purposes).
                    cv.imshow("Original Frame", frame)

                    # Display the processed frame (for debugging purposes).
                    # Note: The processed frame needs to be converted back to a format suitable for display.
                    processed_display_frame = (processed_frame[0] * 255).astype(np.uint8)  # Rescale and convert to uint8
                    processed_display_frame = cv.cvtColor(processed_display_frame, cv.COLOR_RGB2BGR)  # Convert RGB back to BGR
                    cv.imshow("Processed Frame", processed_display_frame)

                    # Break the loop on pressing 'q'.
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        logging.info("Exiting the frame processing pipeline on user request.")
                        break

        except RuntimeError as e:
            logging.error(f"An error occurred: {e}")
        finally:
            # Release resources and close any OpenCV windows.
            logging.info("Releasing resources and closing windows.")
            self.video_stream.release_stream()
            cv.destroyAllWindows()


if __name__ == "__main__":
    # Configure logging for the pipeline.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run the frame processing pipeline.
    pipeline = FramePipeline(capture_device=0, frame_width=1280, frame_height=720, target_width=640, target_height=640)
    pipeline.run()
