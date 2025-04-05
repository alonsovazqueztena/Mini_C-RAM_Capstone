# Daniel Saravia Source:https://grok.com/share/bGVnYWN5_7ddbc9c0-fa9e-43da-8695-e741c0c78579
import logging
from DMX_frame_pipeline import DMXFramePipeline  # Updated import

def main():
    # Configure logging with timestamps and log levels.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize the DMXFramePipeline with the updated parameters.
        pipeline = DMXFramePipeline(
            model_path="drone_detector_12n.pt",
            confidence_threshold=0.5
        )
        # Run the pipeline continuously.
        pipeline.run()

    except Exception as e:
        logging.error(f"DMXFramePipeline execution failed: {e}")

if __name__ == "__main__":
    main()