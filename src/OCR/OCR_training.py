import numpy as np
import time

from src.OCR.OCR_classification import OCRClassifier
from src.training.NSVM_Mangasarian import NSVM_Mangasarian

class OCRTrainer(OCRClassifier):

    def one_versus_all_train(self, A, d, d_tag, output_nsvm=0, wp1=0.5, arm=0):
        """
        Train one-vs-all classifiers using the NSVM Mangasarian algorithm.

        Parameters:
            A (np.ndarray): Training data, shape (n_samples, n_features)
            d (np.ndarray): Class labels, shape (n_samples,)
            d_tag (list or np.ndarray): Tags for classes (not used here but passed for consistency)
            output_nsvm (int): Whether to print NSVM training output
            wp1 (float): Class +1 weight
            arm (int): Use Armijo optimization if set to 1

        Returns:
            classifiers_w (np.ndarray): Weight vectors of shape (n_classes, n_features)
            classifiers_gamma (np.ndarray): Gamma values of shape (n_classes,)
            classes (np.ndarray): Unique class labels
            tags (np.ndarray): Unique tags from d_tag
            cpu_time (float): Total CPU time for training all classifiers
        """

        classes = np.unique(d)
        tags = np.unique(d_tag)

        classifiers_w = []
        classifiers_gamma = []

        start_time = time.process_time()

        for i, class_label in enumerate(classes):
            # Create binary labels for current class vs. all others
            dd = np.where(d == class_label, 1, -1)

            if len(dd) > 0:
                w, gamma, results = NSVM_Mangasarian(A, dd, k=0, nu=0)

                classifiers_w.append(w)
                classifiers_gamma.append(gamma)

        cpu_time = time.process_time() - start_time

        # Convert to numpy arrays
        classifiers_w = np.vstack(classifiers_w)
        classifiers_gamma = np.array(classifiers_gamma)

        return classifiers_w, classifiers_gamma, classes, tags, cpu_time

    def one_versus_all_correctness(self, A, d, classifiers_w, classifiers_gamma, classes, tags):
        """
        Evaluate classification correctness of one-vs-all classifier.

        Parameters:
            A (np.ndarray): Data to classify, shape (n_samples, n_features)
            d (np.ndarray): True class labels, shape (n_samples,)
            classifiers_w (np.ndarray): Weight vectors from trained classifiers, shape (n_classes, n_features)
            classifiers_gamma (np.ndarray): Gamma offsets, shape (n_classes,)
            classes (np.ndarray): Unique class labels
            tags (np.ndarray): Tags corresponding to class labels

        Returns:
            corr (float): Classification accuracy in percentage
        """
        r, c = A.shape
        chosen_class = np.zeros(r)
        chosen_tag = np.zeros(r)

        for pattern_no in range(r):
            class_label, tag_label = self.one_versus_all_classify_point(
                A[pattern_no, :], classifiers_w, classifiers_gamma, classes, tags
            )
            chosen_class[pattern_no] = class_label
            chosen_tag[pattern_no] = tag_label

        correct_count = np.sum(chosen_class == d)
        corr = (correct_count / r) * 100.0

        return corr
