import os
import sys
import time
import numpy as np
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.OCR.OCR_training import OCRTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_LINREC(train_data_filename, test_data_filename, hyperplane_filename):

    # configuration
    wp1 = 0.5
    output_mnsvm = 1
    output_nsvm = 0
    arm = 0
    save_on = True

    # load train data
    train_data = np.load(train_data_filename, allow_pickle=True)
    A_train = train_data['A']
    d_train = train_data['d'].flatten()
    d_train_tag = train_data['d_tag'].flatten()

    # load test data
    test_data = np.load(test_data_filename, allow_pickle=True)
    A_test = test_data['A']
    d_test = test_data['d'].flatten()
    d_test_tag = test_data['d_tag'].flatten()

    # eval hyperplanes
    start_time = time.time()
    classifiers_w, classifiers_gamma, classes, tags, train_corr, test_corr = mnsvm(
        A_train, A_test, d_train, d_test, d_train_tag, d_test_tag,
        output_mnsvm, output_nsvm, wp1, arm
    )
    cpu_time = time.time() - start_time

    # %%%%%%%%%%%%%% save found hyperplane %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if save_on:
        np.savez(hyperplane_filename, {
            'classifiers_w': classifiers_w,
            'classifiers_gamma': classifiers_gamma,
            'classes': classes,
            'tags': tags
        })

    return cpu_time

def mnsvm(A_train, A_test, d_train, d_test, d_train_tag, d_test_tag,
          output_mnsvm=0, output_nsvm=0, wp1=0.5, arm=0):
    """
    Author: Zuzana Petrakova

    Parameters:
        A_train: numpy.ndarray - training set
        A_test: numpy.ndarray - test set
        d_train: numpy.ndarray - training labels
        d_test: numpy.ndarray - testing labels
        d_train_tag: list or array - tags of training labels
        d_test_tag: list or array - tags of testing labels
        output_mnsvm: int (0 or 1) - print MN-SVM results
        output_nsvm: int (0 or 1) - print NSVM results (used internally)
        wp1: float - class 1 weight (0 <= wp1 <= 1)
        arm: int (0 or 1) - use Armijo method if 1

    Returns:
        classifiers_w: list of np.ndarray - weight vectors for binary classifiers
        classifiers_gamma: list of float - bias terms
        classes: list - class labels
        tags: list - tags corresponding to classes
        train_corr: float - training set accuracy
        test_corr: float - test set accuracy
    """

    # Train classifiers in one-vs-all setup
    trainer = OCRTrainer()

    classifiers_w, classifiers_gamma, classes, tags, cpu_time = trainer.one_versus_all_train(
        A_train, d_train, d_train_tag, output_nsvm
    )

    # Evaluate training and testing correctness
    train_corr = trainer.one_versus_all_correctness(A_train, d_train, classifiers_w, classifiers_gamma, classes, d_train_tag)
    test_corr = trainer.one_versus_all_correctness(A_test, d_test, classifiers_w, classifiers_gamma, classes, d_test_tag)

    if output_mnsvm == 1:
        print(f"\nTraining set correctness (sign function): {train_corr:.2f}%")
        print(f"Testing set correctness (sign function): {test_corr:.2f}%")
        print(f"Elapsed time: {cpu_time:.2f} seconds\n")

    return classifiers_w, classifiers_gamma, classes, tags, train_corr, test_corr


def main():
    train_data_filename = r"../../data/LINREC_OCR_TRAINING_DATA_900.npz"
    test_data_filename = r"../../data/LINREC_OCR_TESTING_DATA.npz"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperplane_filename = os.path.join("../../data", f"HYPERPLANE_NSVM_900_{timestamp}.npz")

    train_LINREC(train_data_filename, test_data_filename, hyperplane_filename)

if __name__ == "__main__":
    main()