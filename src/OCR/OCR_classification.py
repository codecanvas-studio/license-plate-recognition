import numpy as np
import time
import logging

class OCRClassifier:
    def __init__(self, classifier_file=None, logger=None):
        self.logger = logger or self._create_default_logger()
        if classifier_file is not None:
            self._load_hyperplane(classifier_file)

    def _load_hyperplane(self, classifier_file):
        # Load hyperplane data
        classifier_data = np.load(classifier_file, allow_pickle=True)
        self.classifiers_w = classifier_data['W']
        self.classifiers_gamma = classifier_data['gamma']
        self.classes = classifier_data['classes']
        self.tags = classifier_data['tags']

        self.logger.info(f"OCRClassifier initialized with {len(self.classes)} classes.")

    def _create_default_logger(self):
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def one_versus_all_classify_point(self, point, classifiers_w,
                                      classifiers_gamma,
                                      classes, tags, ):
        """
        Classifies a single input vector using one-vs-all classification strategy.
        """

        m = len(classes)
        scores = np.zeros(m)

        for i in range(m):
            value = np.dot(point, classifiers_w[i, :].T) - classifiers_gamma[i]
            scores[i] = value

        class_index = np.argmax(scores)
        class_label = classes[class_index]
        tag_label = tags[class_index]

        return class_label, tag_label

    def classify_batch(self, A):
        start_time = time.time()

        r, c = A.shape
        chosen_class = np.zeros(r)
        chosen_tag = np.empty(r, dtype=str)

        self.logger.info(f"Starting OCR classification for {r} patterns with feature length {c}.")

        for pattern_no in range(r):
            class_label, tag_label = self.one_versus_all_classify_point( A[pattern_no, :],self.classifiers_w,self.classifiers_gamma,self.classes, self.tags)
            chosen_class[pattern_no] = class_label
            self.logger.debug(f"Pattern {pattern_no}: Class {class_label}, Tag {tag_label}")

            if isinstance(tag_label, np.ndarray):
                while isinstance(tag_label, np.ndarray):
                    tag_label = tag_label[0]
            chosen_tag[pattern_no] = tag_label

        cpu_time = time.time() - start_time
        self.logger.info(f"OCR classification completed in {cpu_time:.3f} seconds.")
        return chosen_tag, cpu_time



