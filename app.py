import streamlit as st
from PIL import Image
from src.predict import predict_image
from src.model import load_model

st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("Cat vs. Dog Image Classifier")
st.write("Upload an image and let the model classify it.")

model = load_model("models/model.pth", num_classes=2)
CLASS_NAMES = ["cat", "dog"]

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with open("temp.jpg", "wb") as f:
        f.write(uploaded.getbuffer())

    result = predict_image("temp.jpg", model, CLASS_NAMES)

    st.success(f"**Prediction: {result['class']}**")
    st.metric("Confidence", f"{result['confidence']*100:.1f}%")

    st.subheader("All probabilities:")
    for cls, prob in result['all_probs'].items():
        st.progress(prob, text=f"{cls}: {prob*100:.1f}%")