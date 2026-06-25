import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("microbe_model.h5")

classes = [
    "Amoeba",
    "Euglena",
    "Hydra",
    "Paramecium",
    "Rod_bacteria",
    "Spherical_bacteria",
    "Spiral_bacteria",
    "Yeast"
]
st.set_page_config(
    page_title="AquaVision AI",
    layout="centered"
)

st.title("💧 AquaVision AI")
st.subheader(
    "AquaVision AI : Système Intelligent de Détection et de Classification des Microorganismes dans l'Eau"
)

uploaded_file = st.file_uploader(
    "Télécharger une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = image.convert("RGB")

    st.image(
        image,
        caption="Image téléchargée",
        use_container_width=True
    )

    img = image.resize((224,224))
    img = np.array(img)

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(
    f"Microorganisme détecté : {classes[predicted_class]}"
     )

    st.info(
        f"Confiance : {confidence:.2f}%"
    )