# src/run.py
import logging
from DMX_frame_pipeline import FramePipeline

def main():
    # Configure logging to output messages with timestamp and level.
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Initialize the FramePipeline with required parameters.
        pipeline = FramePipeline(
            capture_device=0,
            frame_width=1920,
            frame_height=1080,
            target_width=1920,
            target_height=1080,
            model_path="drone_detector_12x.pt",
            confidence_threshold=0.5
        )
        
        # Run the pipeline continuously.
        pipeline.run()
        
    except Exception as e:
        logging.error(f"FramePipeline execution failed: {e}")

if __name__ == "__main__":
    main()
