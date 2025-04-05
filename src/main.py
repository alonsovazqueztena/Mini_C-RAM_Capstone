# Alonso Vazquez Tena | STG-452: Capstone Project II | April 5, 2025
# Source: https://chatgpt.com/share/67a17189-ca30-800e-858d-aac289e6cb56

import logging # Logging for errors/status.
from frame_pipeline import FramePipeline # Frame pipeline.

def test_frame_pipeline():
    """Test the FramePipeline by processing a video stream with AI detection."""
    try:
        pipeline = FramePipeline() # Initialize pipeline.
        pipeline.run() # Run pipeline until user quits.
    except Exception as e:
        logging.error(f"FramePipeline test failed: {e}") # Log error.
 
def main():
    """Main entry point for testing all modules."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") # Configure logging.
    test_frame_pipeline() # Run frame pipeline test.

if __name__ == "__main__":
    main() # Execute main