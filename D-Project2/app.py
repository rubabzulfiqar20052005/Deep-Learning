import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐶",
    layout="wide"
)

# -----------------------
# LOAD MODEL
# -----------------------

@st.cache_resource
def load_my_model():

    model = tf.keras.models.load_model(
        "best_model.keras"
    )

    return model

model = load_my_model()

# -----------------------
# TITLE
# -----------------------

st.title("🐱 Cats vs Dogs Classification")

st.markdown("""
Deep Learning Project

Models Trained:

- Feed Forward Neural Network
- Basic CNN
- Deep CNN (Best Model)

Upload an image and the trained model will predict
whether it is a Cat or Dog.
""")

# -----------------------
# SIDEBAR
# -----------------------

st.sidebar.header("Project Information")

st.sidebar.write("""
Dataset:
Cats vs Dogs Dataset

Image Size:
128 x 128

Model:
Deep CNN

Output:
Cat / Dog Prediction
""")

# -----------------------
# FILE UPLOAD
# -----------------------

uploaded_file = st.file_uploader(

    "Upload an Image",

    type=["jpg","jpeg","png"]

)

# -----------------------
# PREDICTION
# -----------------------

if uploaded_file:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    img = image.resize(
        (128,128)
    )

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array
    )

    confidence = float(
        prediction[0][0]
    )

    if confidence > 0.5:

        label = "🐶 DOG"

        score = confidence * 100

    else:

        label = "🐱 CAT"

        score = (1-confidence) * 100

    st.success(
        f"Prediction: {label}"
    )

    st.info(
        f"Confidence: {score:.2f}%"
    )