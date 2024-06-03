import streamlit as st
from app1 import run_streamlit_app
import cv2
import pytesseract
import pandas as pd
import numpy as np
import torch
from transformers import BertForTokenClassification, BertTokenizer

# Set page title
st.title('OCR Web App')

# Function to perform OCR
def perform_ocr(image):
    text = pytesseract.image_to_string(image)
    return text

# Streamlit app
def main():
    # Upload image file
    uploaded_file = st.file_uploader("Upload Image 1",key="image1", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Display uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform OCR
        text = perform_ocr(image)
        st.header("Extracted Text:")
        st.write(text)

        # Load pre-trained BERT model for NER
        model = BertForTokenClassification.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Extract structured data using NLP
        tokens = tokenizer.tokenize(text)
        inputs = tokenizer.encode(text, return_tensors="pt")
        outputs = model(inputs)[0]

        # Example: Display processed data in a DataFrame
        st.header("Processed Data:")
        processed_data = pd.DataFrame({'Extracted Text': [text]})
        st.dataframe(processed_data)
        run_streamlit_app()

if __name__ == "__main__":
    main()
