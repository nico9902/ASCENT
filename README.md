# ASCENT \(Attention-based-Slice-Combination-for-survIval-prEdictioN-from-CT\)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Paper](https://img.shields.io/badge/Paper-Springer-blue)](https://doi.org/10.1007/s13755-025-00404-z)

This repository contains the official implementation of the paper:

**"Predicting Lung Cancer Survival with Attention-based CT Slices Combination"**  
Authors: Domenico Paolo, Carlo Greco, Edy Ippolito, Michele Fiore, Sara Ramella, Paolo Soda,
Matteo Tortora, Alessandro Bria, Rosa Sicilia.

The project focuses on predicting the 2-year Overall Survival (OS) of Non-Small Cell Lung Cancer (NSCLC) patients by effectively combining 2D CT slice representations using a soft-attention mechanism.

---

## 📌 Overview
Predicting survival from Computed Tomography (CT) scans is challenging due to the high dimensionality of 3D volumes and limited dataset sizes. This framework addresses these challenges by:
* Using a **pre-trained 2D CNN (EfficientNetB0)** to extract robust features from individual CT slices.
* Implementing a **Soft-Attention Module** that dynamically weighs the importance of each slice, focusing on the most informative anatomical regions.
* Integrating a **DeepHit-based survival analysis** head to estimate the probability of survival over discrete time intervals (up to 24 months).

---

## 🏗 Model Architecture
The pipeline consists of three main components:
1.  **Feature Extractor**: An EfficientNetB0 backbone that processes $N$ slices per patient.
2.  **Attention Mechanism**: A dedicated layer that computes attention scores $\alpha_i$ for each slice, aggregating them into a single context vector.
3.  **Survival Predictor**: A Fully Connected network that outputs the risk distribution across 24 monthly time bins.

![Proposed Method](figures/Method.png)

---

## 📊 Dataset & Preprocessing
The model was validated on the public **LUNG1 (NSCLC-Radiomics)** dataset and a private clinical dataset.

### Preprocessing Pipeline:
* **Resampling**: Voxel spacing standardized to $1 \times 1 \times 3$ mm.
* **Lung Masking**: Slices are filtered based on lung area (threshold > 2%) using a U-Net segmenter.
* **HU Clipping**: Hounsfield Units clipped to $[-1000, 400]$ range.
* **Normalization**: Min-Max scaling and resizing to $224 \times 224$ pixels.

![Preprocessing](figures/Preprocessing.png)

---

## 🎓 Citation

If you use this code, please cite our work:
```
@article{paolo2026predicting,
  title={Predicting lung cancer survival with attention-based CT slices combination},
  author={Paolo, Domenico and Greco, Carlo and Ippolito, Edy and Fiore, Michele and Ramella, Sara and Soda, Paolo and Tortora, Matteo and Bria, Alessandro and Sicilia, Rosa},
  journal={Health Information Science and Systems},
  volume={14},
  number={1},
  pages={20},
  year={2026},
  publisher={Springer}
}
```
---

## 📜 License

This project is licensed. Please review the [LICENSE](LICENSE) file for more information.
