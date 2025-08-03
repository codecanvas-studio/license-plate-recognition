import cv2
import numpy as np
import time
import logging

class PlateDetector:
    """
    A class to detect license plates in vehicle images using image preprocessing,
    component analysis, and projection peak detection.
    """
    def __init__(self, black_thresh=50, car_resize_factor=0.3,
                 threshold_ratio=0.1, min_distance=5,
                 logger=None):
        self.black_thresh = black_thresh
        self.car_resize_factor = car_resize_factor
        self.threshold_ratio = threshold_ratio
        self.min_distance = min_distance
        self.logger = logger or self._create_default_logger()
        self.canny_minval = 100
        self.canny_maxval = 200

    def _create_default_logger(self):
        logger = logging.getLogger('PlateDetector')
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def find_peaks_projection(self, proj):
        threshold = np.max(proj) * self.threshold_ratio
        peaks = []
        for i in range(1, len(proj) - 1):
            if proj[i] > threshold and proj[i] > proj[i - 1] and proj[i] > proj[i + 1]:
                if len(peaks) == 0 or i - peaks[-1] > self.min_distance:
                    peaks.append(i)
        self.logger.debug(f"Found {len(peaks)} peaks.")
        return peaks

    def _preprocess_image(self, image_path):
        RGB_orig = cv2.imread(image_path)
        RGB = cv2.resize(RGB_orig, None, fx=self.car_resize_factor, fy=self.car_resize_factor)

        mask = np.all(RGB < self.black_thresh, axis=2)
        RGB[~mask] = [255, 255, 255]

        gray = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)
        med_filt_size = max(1, int(gray.shape[0] * 0.005) | 1)
        gray = cv2.medianBlur(gray, med_filt_size)

        _, BW = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        BW = cv2.bitwise_not(BW)

        return RGB_orig, RGB, gray, BW

    def _filter_components(self, BW, gray_shape):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(BW, connectivity=8)
        min_w, min_h = int(gray_shape[1] * 0.01), int(gray_shape[0] * 0.01)
        max_w, max_h = int(gray_shape[1] * 0.1), int(gray_shape[0] * 0.1)
        min_area = gray_shape[0] * gray_shape[1] * 0.0001
        max_area = gray_shape[0] * gray_shape[1] * 0.01

        filtered = np.zeros_like(BW)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if (min_area < area < max_area and min_w < w < max_w and min_h < h < max_h):
                ratio = h / w if w != 0 else 0
                if 1 < ratio < 15:
                    filtered[y:y + h, x:x + w] = 255
        return filtered

    def detect_plates(self, image_path, output=False):
        """
        Detect license plate candidates in the given image.

        Args:
                image_path (str): Path to the input image.
                output (bool): If True, display the image with detected plates.

        Returns:
                list of dicts: Each with 'rect' key containing [x, y, w, h] bounding boxes.
                float: Total CPU time for processing.
        """

        BAND_HEIGHT_RATIO = 0.05
        PLATE_WIDTH_RATIO = 0.1

        start_time = time.time()
        RGB_orig, RGB, gray, BW = self._preprocess_image(image_path)
        BR_cropped = self._filter_components(BW, gray.shape)

        edges = cv2.Canny(BR_cropped,  self.canny_minval,  self.canny_maxval)
        VP = np.sum(edges, axis=1)
        smooth_k = max(1, int(gray.shape[0] * 0.005))
        SVP = np.convolve(VP, np.ones(smooth_k) / smooth_k, mode='same')

        peaks = self.find_peaks_projection(SVP)

        plate_candidates = []
        for peak in peaks:
            band_height = int(gray.shape[0] * BAND_HEIGHT_RATIO)
            top = max(peak - band_height, 0)
            bottom = min(peak + band_height, gray.shape[0])
            seg_band = edges[top:bottom, :]

            HP = np.sum(seg_band, axis=0)
            smooth_k_h = max(1, int((bottom - top) * BAND_HEIGHT_RATIO))
            SHP = np.convolve(HP, np.ones(smooth_k_h) / smooth_k_h, mode='same')

            peaks_h = self.find_peaks_projection(SHP)
            for peak_h in peaks_h:
                plate_width = int(gray.shape[1] * PLATE_WIDTH_RATIO)
                left = max(peak_h - plate_width, 0)
                right = min(peak_h + plate_width, gray.shape[1])

                abs_x = int(left / self.car_resize_factor)
                abs_y = int(top / self.car_resize_factor)
                abs_w = int((right - left) / self.car_resize_factor)
                abs_h = int((bottom - top) / self.car_resize_factor)

                plate_candidates.append({'rect': [abs_x, abs_y, abs_w, abs_h]})
                if output:
                    cv2.rectangle(RGB_orig, (abs_x, abs_y), (abs_x + abs_w, abs_y + abs_h), (0, 255, 0), 2)

        if output:
            cv2.imshow('Detected Plates', RGB_orig)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cpu_time = time.time() - start_time
        self.logger.info(f"Plate detection completed in {cpu_time:.2f}s, found {len(plate_candidates)} candidates.")

        return plate_candidates, cpu_time