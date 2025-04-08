# 🧠 Enhanced Lightweight Bearing Defect Detection via Frequency Domain Analysis and Mode

This repository contains the source code and dataset link for the paper:

**"Enhanced Lightweight Bearing Defect Detection via Frequency Domain Analysis and Mode"**  
📌  **The Visual Computer (2025)**

---

## 📌 Introduction

Bearing defect detection plays a crucial role in fault diagnosis and preventive maintenance of industrial equipment. This project proposes a lightweight YOLO-FCMP, an improved lightweight object detection model based on YOLOv7-tiny, specifically designed to identify surface defects in bearings, offering high accuracy and low computational cost.

**Highlights:**
- ⚙️ **FSC Module**: Combines fractional Fourier transform (FrFT), spatial attention, and convolution for enhanced frequency-domain feature representation.
- 🧩 **CAMS Attention**: A novel attention mechanism that improves spatial attention via multi-scale convolutions.
- 📦 **Model Compression**: Uses pruning and knowledge distillation to reduce computation (only 4.6 GFLOPs).
- 🎯 **Performance**: Achieves 99.4% mAP on the surface bearing defect dataset.

---

📊 Results
![image](https://github.com/user-attachments/assets/ef3d2c85-334b-44cc-a93e-3ad20b7efafe)


## 📁 Dataset

The dataset used in this project is **not uploaded directly** due to GitHub’s file size limitation (100MB).  
You can download the dataset via the link below and place it in the `data/` directory.

👉 [**Click here to download the dataset**](https://pan.quark.cn/s/4777df5dac7b)


##  code

compress-1 represents pruning computation.
train-distill refers to the distillation technique.
plot_channel_image represents the comparison image of channels before and after pruning.
