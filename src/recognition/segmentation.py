import time
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlateSegmenter:
    """
       PlateSegmenter is a class used to process and segment individual characters
       from a cropped license plate image.

       It takes a binary image of a license plate as input,
       performs preprocessing, and identifies character-like regions based on
       connected components and geometric filtering.

       Attributes:
           threshold_ratio (float): Ratio of maximum projection value used for peak detection.
           min_distance (int): Minimum distance between peaks in projections.
           logger (logging.Logger): Logger instance for debug/info messages.
    """
    def __init__(self,
                 min_ratio=1.0,
                 max_ratio=15,
                 chars_low=7,
                 chars_up=7,
                 final_height=50,
                 final_width=50,
                 w_diff_coef=2,
                 h_diff_coef=0.3,
                 groups_max_count=4):
        """
            Initialize segmentation parameters:
            - character size constraints
            - aspect ratio limits
            - padding dimensions
            - max allowed outlier grouping iterations
        """
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.chars_low = chars_low
        self.chars_up = chars_up
        self.final_height = final_height
        self.final_width = final_width
        self.w_diff_coef = w_diff_coef
        self.h_diff_coef = h_diff_coef
        self.groups_max_count = groups_max_count

    def read_and_preprocess_image(self, file_path):
        """
            Load image, convert to grayscale, apply blur and binarization (Otsu),
            then invert the binary image to highlight characters.
        """
        RGB = cv2.imread(file_path)
        I = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)

        f_size = max(1, int(I.shape[0] * 0.01))
        if f_size % 2 == 0:
            f_size += 1
        I = cv2.medianBlur(I, f_size)

        _, BW = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        BW = cv2.bitwise_not(BW)
        return RGB, BW

    def extract_connected_components(self, BW):
        """
            Identify connected components in the binary image.
            Filter them by size and aspect ratio constraints.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(BW, connectivity=8)
        components = []
        rows, columns = BW.shape
        min_width = int(columns * 0.01)
        min_height = int(rows * 0.25)
        max_width = int(columns * 0.5)
        max_height = int(rows * 1)

        w = stats[1:, cv2.CC_STAT_WIDTH]
        h = stats[1:, cv2.CC_STAT_HEIGHT]
        x = stats[1:, cv2.CC_STAT_LEFT]
        y = stats[1:, cv2.CC_STAT_TOP]

        r = h / np.maximum(w, 1)
        valid = (h < max_height) & (h > min_height) & (w > min_width) & (w < max_width) & (r < self.max_ratio) & (r > self.min_ratio)
        indices = np.where(valid)[0] + 1

        for i in indices:
            comp_map = np.zeros_like(BW, dtype=np.uint8)
            comp_map[labels == i] = 255
            components.append({
                'map': comp_map,
                'min_x': stats[i, cv2.CC_STAT_LEFT],
                'min_y': stats[i, cv2.CC_STAT_TOP],
                'max_x': stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH],
                'max_y': stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
            })
        return components

    def resize_and_pad_characters(self, components, output_debug=False):
        """
            Resize each character component to uniform height and width, with padding.
        """
        chars = []
        for comp in components:
            char_img = comp['map'][comp['min_y']:comp['max_y'], comp['min_x']:comp['max_x']]
            h, w = char_img.shape
            ratio = self.final_height / h
            new_w = int(round(w * ratio))
            char_resized = cv2.resize(char_img, (new_w, self.final_height), interpolation=cv2.INTER_AREA)

            w_diff = self.final_width - new_w
            if w_diff > 0:
                pad_left = w_diff // 2
                pad_right = w_diff - pad_left
                char_padded = cv2.copyMakeBorder(char_resized, 0, 0, pad_left, pad_right,
                                                 borderType=cv2.BORDER_CONSTANT, value=0)
            else:
                char_padded = char_resized

            char_final = cv2.resize(char_padded, (self.final_width, self.final_height), interpolation=cv2.INTER_NEAREST)

            if output_debug:
                plt.figure()
                plt.imshow(char_final, cmap='gray')
                plt.axis('off')
                plt.show()

            chars.append(char_final)
        return chars

    def segment_plate(self, mode, plate=None, input_file=None, input_directory=None, output_directory=None, output_debug=False):
        """
            Main entry point for license plate segmentation.
            Supports direct image input or file-based mode.
            Returns extracted characters (as images), whether segmentation was successful, and time taken.
        """
        start_time = time.time()
        found = False

        if mode == 'data_acquisition':
            file_path = os.path.join(input_directory, input_file)
            RGB, BW = self.read_and_preprocess_image(file_path)
        else:
            RGB = plate
            I = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)
            f_size = max(1, int(I.shape[0] * 0.01))
            if f_size % 2 == 0:
                f_size += 1
            I = cv2.medianBlur(I, f_size)
            _, BW = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            BW = cv2.bitwise_not(BW)

        components = self.extract_connected_components(BW)
        if not components:
            cpu_time = time.time() - start_time
            logger.info("No components found.")
            return [], found, cpu_time

        comp_group = [{'input': components, 'input_count': len(components)}]
        group_index = 0

        while True:
            group = comp_group[group_index]
            widths = np.array([c['max_x'] - c['min_x'] for c in group['input']])
            heights = np.array([c['max_y'] - c['min_y'] for c in group['input']])

            median_width = np.median(widths)
            median_height = np.median(heights)

            diff_width = median_width * self.w_diff_coef
            diff_height = median_height * self.h_diff_coef

            keep_mask = ((heights > (median_height - diff_height)) & (heights < (median_height + diff_height)) &
                         (widths > (median_width - diff_width)) & (widths < (median_width + diff_width)))

            output_group = [group['input'][i] for i, keep in enumerate(keep_mask) if keep]
            next_input = [group['input'][i] for i, keep in enumerate(keep_mask) if not keep]

            comp_group[group_index]['output'] = output_group
            comp_group[group_index]['output_count'] = len(output_group)

            if len(next_input) == 0 or group_index + 1 >= self.groups_max_count or len(output_group) == 0:
                break

            comp_group.append({'input': next_input, 'input_count': len(next_input)})
            group_index += 1

        ok_char_count = [i for i, g in enumerate(comp_group) if self.chars_low <= g['output_count'] <= self.chars_up]

        if not ok_char_count:
            cpu_time = time.time() - start_time
            logger.info("No valid group found with appropriate character count.")
            return [], found, cpu_time

        if len(ok_char_count) > 1:
            median_heights = [np.median([c['max_y'] - c['min_y'] for c in comp_group[i]['output']]) for i in ok_char_count]
            chosen_group = ok_char_count[np.argmax(median_heights)]
        else:
            chosen_group = ok_char_count[0]

        found = True

        sorted_components = sorted(comp_group[chosen_group]['output'], key=lambda c: c['min_x'])
        chars = self.resize_and_pad_characters(sorted_components, output_debug=output_debug)

        cpu_time = time.time() - start_time
        logger.info(f"Segmentation completed in {cpu_time:.3f} seconds. Found: {found}. Characters: {len(chars)}")

        return chars, found, cpu_time
