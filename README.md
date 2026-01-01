# HYPERSPECTRAL-ANOMALY-DETECTION
## Hyperspectral Anomaly Detection using LRSR

Hyperspectral Anomaly Detection using Dictionary Learning, Low-Rank and Sparse Representation (LRSR). The project constructs background and anomaly dictionaries from hyperspectral images, performs sparse coding with SOMP, detects anomalies via low-rank and sparse decomposition, visualizes results, and evaluates performance using ROC-AUC.

---

## Features
- Preprocessing of hyperspectral images including band removal, normalization, and PCA dimensionality reduction.
- Background and anomaly dictionary construction using local windows and clustering.
- Sparse coding using Simultaneous Orthogonal Matching Pursuit (SOMP).
- Low-rank and sparse decomposition for anomaly detection.
- Visualization of background, anomaly, noise, dictionary maps, and segmentation results.
- ROC-AUC evaluation of detection performance.

---

## Requirements
- Python 3.x
- Libraries:
  - numpy
  - scipy
  - scikit-learn
  - matplotlib

Install dependencies using:
```bash
pip install numpy scipy scikit-learn matplotlib

---

File Structure

main.py - Entry point for data loading, dictionary construction, LRSR computation, visualization, and ROC-AUC evaluation.

dictionary.py - Functions to build background and anomaly dictionaries from hyperspectral images.

lrsr_.py - Low-Rank and Sparse Representation computations.

visualization.py - Functions to display results and compute ROC-AUC.

utils.py - Helper functions for data conversion, normalization, SOMP, and window creation.
