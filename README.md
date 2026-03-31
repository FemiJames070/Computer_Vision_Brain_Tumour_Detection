# 🧠 AI-Based Brain Tumor Detection: Clinical Computer Vision

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io/)

## 📖 Project Overview
This repository contains an end-to-end Computer Vision pipeline designed to automatically detect brain tumors from MRI scans. Built using the **CRISP-DM** methodology, this project leverages Deep Learning and **Transfer Learning (Xception Architecture)** to serve as an AI-assisted clinical decision-support tool, improving diagnostic speed and reliability.

**Dataset:** [Brain MRI Images Dataset (Mendeley Data)](https://data.mendeley.com/datasets/c9rt8d6zrf/1) (~10,093 images)

## 🎯 Business Value & Objectives
Brain tumor diagnosis is time-sensitive and highly dependent on radiologist expertise. This project bridges the gap between raw medical imaging and operational healthcare delivery by:

1. **Reducing Diagnostic Fatigue:** Providing a highly reliable "first-pass" screening tool to assist medical professionals.
2. **Minimizing False Negatives:** Optimizing the model for high **Recall (Sensitivity)**, ensuring that actual tumors are rarely missed (only a ~3.5% False Negative rate).
3. **Maximizing Efficiency:** Utilizing Transfer Learning to achieve state-of-the-art accuracy without the massive computational overhead of training a Deep CNN from scratch.

## ⚙️ The Architecture: Where Vision Happens

While NLP models are fundamentally designed to guess the next word in a sequence, this Computer Vision architecture is built to mathematically deconstruct, interpret, and understand the physical world. It requires a higher degree of spatial multidimensionality. The raw code is available in the repository, but the strategic philosophy behind the model building is what makes it highly effective.

### 1. The Training Crucible & Data Forging
A vision model is only as robust as the environment it’s trained in. We didn't just feed the network data; we aggressively transformed it.
* **Perfectly Balanced Dataset:** The dataset contains 10,093 images with a near 50/50 split between "Tumor" and "No Tumor," eliminating the need for synthetic oversampling (SMOTE).
* **Aggressive Augmentation:** Applied random horizontal flips, rotations (10%), and zooming (10%) on the fly. By forcing the model to interpret augmented realities, we ensured it learned true spatial invariance across different MRI cross-sections (Axial, Sagittal, Coronal) rather than just memorizing pixel patterns.
* **Pipeline Optimization:** Utilized `tf.data.AUTOTUNE`, caching, and prefetching to eliminate I/O bottlenecks during GPU training.

### 2. Strategic Model Building (Xception)
Rather than throwing layers at a GPU and hoping for convergence, we utilized the **Xception** model (pre-trained on ImageNet) as our architectural foundation:
* **The Frozen Backbone:** We locked the 20.8 million parameters of the base model to act as a highly optimized feature extractor, immediately capable of detecting low-level edges, contrasts, and hyperintense masses without retraining.
* **The Custom Classification Head:** We engineered a custom top layer utilizing Global Average Pooling and aggressive Dropout (0.2) to ruthlessly prune weak connections and prevent overfitting. It concludes with a Dense Sigmoid layer for precise binary classification.
* **High-Efficiency Convergence:** By freezing the backbone, only **2,049 parameters** were actually trained, allowing for rapid convergence while maintaining a lean computational footprint.

## 📈 Model Performance & Results

In advanced Computer Vision, a flat "accuracy" percentage is a vanity metric. The architecture was rigorously evaluated against a strictly unseen test set of 1,816 images, focusing on clinical implications:

| Metric | Score | Clinical Implication |
| :--- | :---: | :--- |
| **Test Accuracy** | **96.81%** | Highly reliable overall classification capability. |
| **Recall (Tumor)** | **0.96** | Caught 96% of all actual tumors (Low False Negative Rate). |
| **Precision (Tumor)**| **0.97** | When the model flags a tumor, it is correct 97% of the time. |
| **F1-Score** | **0.97** | Perfectly balanced model; no bias towards the healthy class. |

*Note: The model successfully distinguished complex healthy anatomical structures (like eye sockets and ventricles) from pathological masses, proving its robustness against critical false positives.*

## 🚀 Future Roadmap
To transition this prototype into a deployment-ready clinical application, the next phases include:

1. **Explainable AI (XAI):** Implementing **Grad-CAM** heatmaps to visually highlight the exact region of the MRI driving the model's prediction, building crucial physician trust.
2. **Multiclass Segmentation:** Expanding the architecture to classify specific tumor types (e.g., Glioma, Meningioma, Pituitary).
3. **Flask Web Deployment:** Wrapping the `.keras` model in a secure, custom **Flask** application for easy MRI uploads and real-time inference, providing a more robust architectural foundation than standard dashboarding tools.

## 🛠️ Technology Stack
* **Language:** Python
* **Deep Learning:** TensorFlow 2.x, Keras
* **Computer Vision:** OpenCV, Matplotlib (for visualization)
* **Data Processing:** NumPy, Scikit-Learn

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/brain-tumor-detection-cv.git](https://github.com/yourusername/brain-tumor-detection-cv.git)
   cd brain-tumor-detection-cv

2. **Install the required dependencies:**
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn seaborn

3. **Run the Notebook:**
Open the Jupyter Notebook to view the step-by-step data pipeline, model training, and visual evaluations.

⚠️ Medical Disclaimer
For Educational and Research Purposes Only. This artificial intelligence model is designed for portfolio demonstration and academic research. It has not been approved by the FDA or any regulatory body and is not intended to be a substitute for professional medical advice, diagnosis, or treatment.

## ✍️ Author
Femi James
Data & Business Analyst | Integrated AI Specialist
