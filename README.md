---
title: AquaVision AI
sdk: docker
app_port: 7860
---
# AquaVision AI

## Intelligent Detection and Classification of Microorganisms in Water Samples

### Technologies Used

#### Frontend

* Streamlit

#### Backend

* Python 3.11
* TensorFlow 2.21
* Keras
* NumPy
* Pillow

#### Artificial Intelligence

* MobileNetV2 (Pretrained Model)
* Transfer Learning
* Data Augmentation
* Dropout

### Dataset

* Total Images: 789
* Training Images: 633
* Validation Images: 156
* Number of Classes: 8

Classes:

* Amoeba
* Euglena
* Hydra
* Paramecium
* Rod_bacteria
* Spherical_bacteria
* Spiral_bacteria
* Yeast

### Model Architecture

MobileNetV2
→ GlobalAveragePooling2D
→ Dense(128, ReLU)
→ Dropout(0.5)
→ Dense(8, Softmax)

### Results

* Training Accuracy: 69.8%
* Validation Accuracy: 66.0%

### Run Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Project Files

```text
app.py
train_model.ipynb
microbe_model.h5
requirements.txt
README.md
```
