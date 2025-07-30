import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# 📌 تحميل النموذج المحفوظ
model = tf.keras.models.load_model("saved_model_arabic_handwriting")

# 📌 تحميل الـ LabelBinarizer
lb = joblib.load("label_binarizer.pkl")

# 📌 تحميل ملف CSV (لازم يكون صف واحد يمثل صورة بحجم 32x32)
csv_file = "D:\Arabic Handwritten Recognition\Arabic Handwritten Characters Dataset CSV\csvTestImages 3360x1024.csv"  # ← عدلي الاسم حسب الملف اللي عندك
image_data = pd.read_csv(csv_file, header=None).values.astype('float32') / 255.0

# ✅ التأكد إن عدد القيم 1024 (يعني 32x32)
if image_data.shape[1] != 1024:
    raise ValueError("❌ الملف يجب أن يحتوي على صورة واحدة مفلطحة بـ 1024 قيمة (32x32).")

# 🟡 إعادة تشكيل البيانات
image = image_data.reshape(-1, 32, 32, 1)

# 🔍 التنبؤ باستخدام النموذج
prediction = model.predict(image)
predicted_class = lb.classes_[np.argmax(prediction)]

# ✅ عرض النتيجة
print("✅ Predicted Character:", predicted_class)

# 🖼️ عرض الصورة
plt.imshow(image[0].reshape(32, 32), cmap="gray")
plt.title(f"Predicted: {predicted_class}")
plt.axis("off")
plt.show()

# 📊 عرض الـ Probabilities لأعلى 5 احتمالات
probs = prediction[0]
top_indices = probs.argsort()[-5:][::-1]
for idx in top_indices:
    print(f"{lb.classes_[idx]}: {probs[idx]*100:.2f}%")
