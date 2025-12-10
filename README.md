# Medical Image Processing Exercises

This repository contains a single Google Colab notebook that walks through **nine image-processing and deep learning exercises** using common medical-imaging style datasets (PNG/JPG, DICOM, segmentation masks, and X-ray images).

The code is written to run in **Google Colab** and assumes that several dataset ZIP files are stored in **Google Drive**.

---

## Contents

All exercises live in one notebook:

- `medical_image_processing_steps.ipynb`

Each exercise builds on core topics in medical image analysis:

1. **Exercise 1 – Basic Image I/O (PNG/JPG & DICOM)**
2. **Exercise 2 – Direction Classification with ResNet50 (Keras)**
3. **Exercise 3 – Gender Classification with ResNet18 (PyTorch)**
4. **Exercise 4 – Age Regression from X-ray Images**
5. **Exercise 5 – Lung Segmentation with a U-Net**
6. **Exercise 6 – Organ Region Extraction From Segmentation Masks**
7. **Exercise 7 – Bounding-Box Localization on Original Images**
8. **Exercise 8 – Autoencoder for Anomaly / Reconstruction Error Detection**
9. **Exercise 9 – Unsupervised Clustering with ResNet Features + K-Means**

---

## Datasets

The notebook expects the following ZIP archives in miniJSRT_database:

`http://imgcom.jsrt.or.jp/minijsrtdb/`

Required files:

- `Practice_PNGandJPG.zip`
- `Practice_DICOM.zip`
- `Directions01.zip`
- `Gender01.zip`
- `XPAge01_RGB.zip`
- `Segmentation01.zip`
- `segmentation02.zip`
- `autoencoder_img.zip`  

> If you use a different folder structure, **update the paths** in the notebook accordingly.

---

## Environment & Dependencies

The notebook is designed for **Google Colab** with a **GPU** runtime.

Main libraries used:

- **Core Python**
  - `zipfile`, `os`, `glob`, `random`, `pathlib`, `matplotlib`, `numpy`, `pandas`
- **Image I/O & Display**
  - `PIL` (`Pillow`)
  - `cv2` (**OpenCV**)
  - `pydicom`
  - `IPython.display`
- **Deep Learning**
  - **PyTorch** (`torch`, `torchvision`)
  - **TensorFlow / Keras** (`tensorflow`, `tf.keras`)
- **Augmentation & Segmentation**
  - `albumentations` (installed in the notebook)
- **ML / Evaluation**
  - `scikit-learn` (metrics, clustering, PCA)
  - `tqdm` for progress bars

Colab-specific installs are included in the notebook, for example:

```python
# Example
!pip install albumentations==1.3.0 -q
# %pip install pydicom  # (commented but can be used if needed)
