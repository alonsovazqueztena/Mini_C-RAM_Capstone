# Daniel Saravia Source:https://grok.com/share/bGVnYWN5_7ddbc9c0-fa9e-43da-8695-e741c0c78579
# run.py
import logging
from DMX_frame_pipeline import DMXFramePipeline  # Updated import

def main():
    # Configure logging with timestamps and log levels.
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG to show all log messages.
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        pipeline = DMXFramePipeline(
            model_path="drone_detector_12x.pt",
            confidence_threshold=0.5
        ) # Initialize DMX pipeline with model path and confidence threshold.
        pipeline.run() # Run the pipeline until user quits.

    except Exception as e:
        logging.error(f"DMXFramePipeline execution failed: {e}")

if __name__ == "__main__":
    main()
