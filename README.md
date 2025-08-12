# LINREC — Linear OCR Pipeline (NSVM-based)

This repository contains an implementation of the LINREC character recognition system, including preprocessing, classification, and training modules based on normalized support vector machines (NSVM).

> ⚠️ This project is part of my personal AI/Python portfolio. It is **not intended for reuse, commercial, or derivative purposes**.

---

## 📌 Project Overview

LINREC is a modular OCR system inspired by my academic work, designed to process segmented character images, extract features, and classify characters using a linear one-vs-all SVM approach.

Key components include:
- Plate detection
- Character segmentation & preprocessing
- Feature extraction (character matrices)
- One-vs-all classifier training using NSVM
- Syntax control

---


## 🚫 License and Usage

This repository is shared under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0** license.

- **You may view or share this repository for personal, educational, or hiring purposes.**
- **You may not** use this code in commercial products, modify it for your own projects, or redistribute modified versions.

See the [`LICENSE`](LICENSE) file for full terms.

If you're interested in learning more about the code, feel free to reach out.

---

## 📁 Structure

license-plate-recognition/
│
├── src/ # All main modules and logic
├── data/ # Training and Testing data
├── main.py # Linrec pipeline for recognition
├── README.md
├── LICENSE
└── requirements.txt


---

## 🔧 Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Scikit-learn (if kernel SVM is used)
- PIL (optional for image operations)


---

## Recogniton

To run license plate recognition on a folder of test images using a pre-trained NSVM model:

- Prepare your data
Place the input images into a folder, e.g., data/test/.

Ensure you have a trained model file, e.g., HYPERPLANE_NSVM_600.npz, saved in the data/ directory.

- Run the script
Execute the main script: main.py
This will:

Load the model from data/HYPERPLANE_NSVM_600.npz
Process all images in data/test/
Save output (such as processed images or logs) to data/test_out/
The console will show recognition progress and results

---

## Training

NSVM Classifier Training Summary
To evaluate the impact of training set size on the accuracy of the NSVM classifier, I conducted experiments 
using three progressively larger training datasets (322, 644, and 956 character images). 
The test set remained fixed at 321 characters—the same size as the smallest training set. 
Across all configurations, the model achieved 100% training accuracy. 
Test accuracy improved slightly with dataset size, reaching up to 99.03% for the two larger sets.