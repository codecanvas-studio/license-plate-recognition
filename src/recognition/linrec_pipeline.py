import os
import time
import numpy as np
import cv2
import logging

from src.OCR.OCR_classification import OCRClassifier
from src.recognition.detection import PlateDetector
from src.recognition.segmentation import PlateSegmenter
from src.recognition.syntax_check import syntax_check

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class LINRECPipeline:
    """
    License plate recognition pipeline that processes a folder of input images,
    detects plates, segments characters, runs OCR, and saves results and timing stats.
    """
    def __init__(self, model_path):
        """
        Initialize the pipeline with a path to the OCR model.
        """
        self.model_path = model_path
        self.classifier = OCRClassifier(self.model_path)
        self.segmenter = PlateSegmenter()

    def process_folder(self, input_images_dir, output_images_dir):
        """
        Process all images in the input directory and save results in the output directory.
        """

        if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)

        images = [f for f in os.listdir(input_images_dir) if not f.startswith('.')]
        results = []

        cpu_time_segment_plate_total = 0
        cpu_time_plate_detection_total = 0
        cpu_time_OCR_classify_total = 0

        for idx, file_name in enumerate(images, start=1):
            logger.info(f"Processing {idx}/{len(images)}: {file_name}")
            start_time = time.time()

            image_path = os.path.join(input_images_dir, file_name)
            RGB_image = cv2.imread(image_path)

            black_thresh = 20
            recognized = False

            for attempt in range(5):
                logger.info(f"Attempt {attempt + 1}: black_thresh={black_thresh}")

                detector = PlateDetector(black_thresh)
                candidates, cpu_time_plate_detection = detector.detect_plates(image_path, False)
                cpu_time_plate_detection_total += cpu_time_plate_detection

                for candidate in candidates:
                    x, y, w, h = candidate['rect']
                    plate = RGB_image[y:y + h, x:x + w]

                    segmenter = PlateSegmenter()
                    chars, found, cpu_time_segment_plate = segmenter.segment_plate('LINREC', plate)
                    cpu_time_segment_plate_total += cpu_time_segment_plate

                    if not found:
                        continue

                    A = np.array([
                        cv2.resize(char, (50, 50), interpolation=cv2.INTER_NEAREST).flatten()
                        for char in chars
                    ])

                    classifier = OCRClassifier(self.model_path)
                    license_number, cpu_time_OCR_classify = classifier.classify_batch(A)
                    cpu_time_OCR_classify_total += cpu_time_OCR_classify

                    logger.info(f"Raw license number: {license_number}")

                    correct, license_number_new = syntax_check(license_number)
                    if correct:
                        recognized = True

                        record = {
                            'name': file_name,
                            'license_number': license_number_new,
                            'cpu_time_plate_detection': cpu_time_plate_detection,
                            'cpu_time_segment_plate': cpu_time_segment_plate,
                            'cpu_time_OCR_classify': cpu_time_OCR_classify,
                            'total_time': time.time() - start_time
                        }
                        results.append(record)

                        output_path = os.path.join(output_images_dir, f"{file_name}.npz")
                        np.savez(output_path, to_save=record)
                        logger.info(f"Recognition successful: {license_number_new}")
                        break

                if recognized:
                    break

                black_thresh += 10

        # Save timing summary
        timing_path = os.path.join(output_images_dir, "time.npz")
        np.savez(timing_path,
                 cpu_time_segment_plate_total=cpu_time_segment_plate_total,
                 cpu_time_plate_detection_total=cpu_time_plate_detection_total,
                 cpu_time_OCR_classify_total=cpu_time_OCR_classify_total)

        logger.info("Batch processing completed.")
        return results
