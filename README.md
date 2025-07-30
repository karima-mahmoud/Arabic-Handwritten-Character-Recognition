# 🧠 Arabic Handwritten Character Recognition with CNN & Streamlit

This project builds a Convolutional Neural Network (CNN) model to classify **handwritten Arabic characters** from grayscale images using the [Arabic Handwritten Character Dataset](). It also includes a **Streamlit web app** for live prediction.

---

## 📌 Features

- ✅ Trainable CNN model on grayscale images  
- 📊 Evaluation with accuracy, precision, recall, F1-score, and ROC curves  
- 📉 Visualization of training/validation performance  
- ⚠️ Misclassification display  
- 🌐 Live inference with Streamlit  
- 🧠 Supports 28 Arabic characters

---

## 🗂️ Dataset

We used the **Arabic Handwritten Characters Dataset (CSV version)**:

- `csvTrainImages 13440x1024.csv` 
- `csvTrainLabel 13440x1.csv` 
- `csvTestImages 3360x1024.csv` 
- `csvTestLabel 3360x1.csv` 

Each row = 1024 pixel values (flattened 32×32 image). Labels are integers (0–27), representing Arabic characters.

---

## 🛠️ Libraries & Tools

- Python, Pandas, NumPy  
- TensorFlow / Keras  
- Matplotlib, Seaborn  
- Scikit-learn  
- Streamlit  
- Joblib

---

## 📈 Model Architecture

- Conv2D (32 filters) → MaxPooling  
- Conv2D (64 filters) → MaxPooling  
- Flatten → Dense(128) → Dropout(0.5)  
- Output Layer: Softmax over 28 classes

---

## 🧪 Evaluation

### ✅ Metrics:
- Accuracy: `91.04%`
- Precision, Recall, F1-score (per class)
- ROC Curve for top 5 classes

### 📊 Visualization:

| Training & Validation Accuracy  |

| ![acc_](<img width="619" height="481" alt="image" src="https://github.com/user-attachments/assets/96a6da7c-2a8b-4ccc-99cd-0f7248016f7e" />
>

#
---

## 🌐 Streamlit Web App

An interactive app that accepts a **.csv file** of a handwritten character and returns the predicted letter + probabilities.

<img width="565" height="694" alt="image" src="https://github.com/user-attachments/assets/cd2420a6-224a-4737-ad8d-4118b120f018" />


### 🎯 How to Run

```bash
streamlit run app.py
