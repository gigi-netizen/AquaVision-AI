import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="AquaVision AI",
    page_icon="💧",
    layout="centered"
)

# CSS personnalisé
st.markdown("""
<style>
.main {
    background: linear-gradient(to bottom, #e0f7fa, #ffffff);
}
h1 {
    text-align: center;
    color: #0077b6;
    font-size: 42px;
}
h3 {
    text-align: center;
    color: #023e8a;
}
.upload-box {
    border: 2px dashed #0077b6;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    background-color: #f1faff;
}
.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #caf0f8;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Charger le modèle
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

# Header
st.title("💧 AquaVision AI")
st.subheader(
    "Système Intelligent de Détection et de Classification des Microorganismes dans l'Eau"
)

# Upload
st.markdown('<div class="upload-box">📤 Importez une image microscope</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Image téléchargée",
        use_container_width=True
    )

    # Prétraitement
    img = image.resize((224, 224))
    img = np.array(img)

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prédiction
    with st.spinner("Analyse en cours..."):
        prediction = model.predict(img)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Résultats
    st.markdown(
        f'<div class="result-box">🔬 Microorganisme détecté : {classes[predicted_class]}</div>',
        unsafe_allow_html=True
    )

    st.progress(int(confidence))

    st.info(f"Confiance : {confidence:.2f}%")
