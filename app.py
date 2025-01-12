import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")

# Function to classify text
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    return probabilities

# Streamlit app
st.set_page_config(page_title="Hate Speech Classifier", layout="centered")

st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f0f5;
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background: #f0f0f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hate Speech Classifier")
st.markdown("### Enter a text to classify whether it's hate speech or not.")

user_input = st.text_area("Text input", "Type your text here...", height=200)

if st.button("Classify"):
    with st.spinner("Classifying..."):
        probabilities = classify_text(user_input)
        st.write("### Results")
        st.write(f"**Not Hate Speech:** {probabilities[0]:.2%}")
        st.write(f"**Hate Speech:** {probabilities[1]:.2%}")

st.markdown("---")
st.markdown("**Note:** This model may not be 100% accurate. Always use caution and human judgment when dealing with sensitive content.")