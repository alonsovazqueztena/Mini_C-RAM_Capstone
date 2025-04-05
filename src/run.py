# src/run.py
import logging
from DMX_frame_pipeline import FramePipeline

def main():
    # Configure logging with timestamps and log levels.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize the DMX FramePipeline with the updated parameters.
        pipeline = FramePipeline(
            model_path="drone_detector_12n.pt",
            confidence_threshold=0.5
        )
        # Run the pipeline continuously.
        pipeline.run()

    except Exception as e:
        logging.error(f"FramePipeline execution failed: {e}")

if __name__ == "__main__":
    main()

