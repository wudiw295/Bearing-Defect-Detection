# 🧠 Bearing Defect Detection Based on Lightweight YOLO-FCMP

This repository contains the source code and dataset link for the paper:

**"YOLO-FCMP: A Lightweight Defect Detection Network for Bearing Surface Based on Multi-dimensional Attention and Feature Compression"**  
📌 *Published in* **The Visual Computer (2025)**

---

## 📌 Introduction

Bearing defect detection plays a crucial role in the fault diagnosis and preventive maintenance of industrial equipment. This project presents **YOLO-FCMP**, an improved lightweight object detection model based on YOLOv7-tiny, specifically designed for identifying surface defects in bearings with high accuracy and low computational cost.

**Highlights:**
- ⚙️ **FSC Module**: Combines fractional Fourier transform (FrFT), spatial attention, and convolution for enhanced frequency-domain feature representation.
- 🧩 **CAMS Attention**: A novel attention mechanism that improves spatial attention via multi-scale convolutions.
- 🌀 **Deformable Convolution (DCNv2)**: Improves detection for irregular and complex defect shapes.
- 📦 **Model Compression**: Uses pruning and knowledge distillation to reduce computation (only 4.6 GFLOPs).
- 🎯 **Performance**: Achieves 99.4% mAP on the surface bearing defect dataset.

---

## 📁 Dataset

The dataset used in this project is **not uploaded directly** due to GitHub’s file size limitation (100MB).  
You can download the dataset via the link below and place it in the `data/` directory.

👉 [**Click here to download the dataset**](https://your-dataset-link.com)  
*(Replace with your actual Google Drive or other permanent link)*

Once downloaded, unzip it as follows:

```bash
cd data/
unzip data.zip
