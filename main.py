import os
import logging
from src.recognition.linrec_pipeline import LINRECPipeline

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    """
    Runs the LINRECPipeline on a folder of input images and logs the recognition results.
    """
    input_images_dir = os.path.join("data", "test")
    output_images_dir = os.path.join("data", "test_out")
    model_path = r"data/HYPERPLANE_NSVM_600.npz"

    # Check that paths exist
    if not os.path.exists(input_images_dir):
        logging.error(f"Input image directory not found: {input_images_dir}")
        return

    if not os.path.exists(model_path):
        logging.error(f"OCR model file not found: {model_path}")
        return

    try:
        pipeline = LINRECPipeline(model_path)

        logging.info("Starting LINREC batch processing...")
        results = pipeline.process_folder(input_images_dir, output_images_dir)
        logging.info("Processing finished.")

        # Log recognition results
        logging.info("Results summary:")
        for res in results:
            logging.info(res)

    except Exception as e:
        logging.exception("Unexpected error during processing.")

if __name__ == "__main__":
    main()
