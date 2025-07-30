# Arabic Handwritten Character Recognition - Streamlit Deployment (Using Saved Model)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from PIL import Image

st.title("🔤 Arabic Handwritten Character Recognition")

# Upload CSV Image File
uploaded_file = st.file_uploader("Upload CSV File of Arabic Handwritten Characters (32x32 image)", type="CSV")

if uploaded_file is not None:
    # Load and preprocess image
    image_data = pd.read_csv(uploaded_file, header=None).values.astype('float32') / 255.0
    if image_data.shape[1] != 1024:
        st.error("The input CSV must contain a 32x32 image flattened to 1024 values.")
    else:
        image = image_data.reshape(-1, 32, 32, 1)

        # Load model and label binarizer
        model = tf.keras.models.load_model("saved_model_arabic_handwriting")
        lb = joblib.load("label_binarizer.pkl")

        # Predict
        prediction = model.predict(image)
        predicted_class = lb.classes_[np.argmax(prediction)]
        st.success(f"✅ Predicted Character: {predicted_class}")

        # Show image
        st.image(image[0].reshape(32, 32), caption="Uploaded Character", width=150, clamp=True)

        # Show probabilities
        st.subheader("Prediction Probabilities")
        probs_df = pd.DataFrame(prediction, columns=lb.classes_)
        st.dataframe(probs_df.T.sort_values(by=0, ascending=False).head(5))
else:
    st.info("📌 Please upload a CSV file containing one or more 32x32 handwritten characters to classify.")
