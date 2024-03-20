
import streamlit as st
import tensorflow as tf
import numpy as np
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import joblib
import matplotlib.pyplot as plt

model=joblib.load(open('/content/Covid_classification.pkl','rb'))

st.title('CNN Model for Covid Classification')

def set_bg_hack_url():    
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://cdn.pixabay.com/photo/2020/05/15/18/46/coronavirus-5174671_1280.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()

# File uploader for image selection
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = plt.imread(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_resized = resize(img, (150, 150, 1))
    img_reshaped = img_resized.reshape(1, 150, 150, 1)

    # Make prediction
    y_new = model.predict(img_reshaped)
    ind = np.argmax(y_new)

    # Display the predicted class
    categories = ['Covid', 'Normal', 'Viral Pneumonia']

    st.write('Predicted Class:', categories[ind])



