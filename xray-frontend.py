import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load the trained model (chest_xray_prediction.h5)
model = load_model("chest-xray.h5")

# Set page configuration and custom CSS for styling
st.set_page_config(page_title="Chest X-ray Pneumonia Detection", layout="wide")

# Apply custom styling to enhance the appearance
st.markdown("""
    <style>
    body {
        background-color: #4B0082; /* Dark purple */
        color: #FFFFFF; /* White text for readability */
    }
    .stButton>button {
        background-color: #9370DB; /* Lighter purple */
        color: white;
        border-radius: 12px;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #FFD700; /* Gold for headings */
        font-family: 'Arial';
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.title('🩺 Chest X-ray Pneumonia Detection')

# Subtitle
st.write('Upload a Chest X-ray image to predict if it\'s Normal or Pneumonia.')

# File uploader to upload images
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

# Check if an image file is uploaded
if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=True)

    # Preprocessing the image
    image = np.array(image)
    resized_image = cv2.resize(image, (224, 224))
    resized_image = resized_image / 255.0  # Normalize the image

    # Expand the dimensions to add the batch size (1, 224, 224, 3)
    resized_image = np.expand_dims(resized_image, axis=0)

    # Predict using the loaded model
    prediction = model.predict(resized_image)

    # Display prediction result
    if prediction > 0.5:
        st.subheader("🛑 Prediction: Pneumonia")
        # Provide medical advice or a note for Pneumonia case
        st.markdown("""
            **Note:** The model predicts Pneumonia. Please consult a healthcare professional for a detailed diagnosis and treatment plan. 
            Early detection and treatment are crucial for managing pneumonia effectively.
        """)
    else:
        st.subheader("✅ Prediction: Normal")
        # Provide medical advice or a note for Normal case
        st.markdown("""
            **Note:** The model predicts the X-ray is normal. If you're experiencing symptoms, it is recommended to seek professional medical advice to ensure accurate health evaluation.
        """)
else:
    st.write("Please upload an image to make a prediction.")

# Footer section for showing "Ume Habiba"
st.markdown("""
    <div class="footer">
        <p>Developed by <strong>Ume Habiba</strong></p>
    </div>
""", unsafe_allow_html=True)
