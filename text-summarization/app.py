import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Constants
MODEL_PATH = "models/t5_summarization"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

st.set_page_config(
    page_title="Finetuned Text Summarizer",
    page_icon="📝",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the model and tokenizer (cached by Streamlit)"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    return tokenizer, model

def generate_summary(text, model, tokenizer, max_length=128, min_length=30):
    """Generate summary for the input text"""
    # Prepare input text
    input_text = "summarize: " + text.strip()
    
    # Tokenize input
    inputs = tokenizer(input_text, max_length=MAX_INPUT_LENGTH, truncation=True, 
                      return_tensors="pt", padding=True)
    
    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    
    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Streamlit UI
st.title("Text Summarizer")
st.write("""
Enter your text below and get an AI-generated summary using a fine-tuned T5 model.
The model has been trained on news articles and can generate concise summaries while maintaining key information.
""")

# Load model
with st.spinner("Loading the model... This might take a few seconds."):
    tokenizer, model = load_model()

# Text input
text_input = st.text_area("Enter the text you want to summarize:", 
                         height=200,
                         placeholder="Paste your article or text here...")

# Advanced settings in sidebar
st.sidebar.header("Settings")
max_length = st.sidebar.slider("Maximum summary length", 30, 150, 128)
min_length = st.sidebar.slider("Minimum summary length", 10, 100, 30)

# Generate summary button
if st.button("Generate Summary"):
    if not text_input.strip():
        st.error("Please enter some text to summarize!")
    else:
        with st.spinner("Generating summary..."):
            summary = generate_summary(
                text_input, 
                model, 
                tokenizer, 
                max_length=max_length,
                min_length=min_length
            )
            
        st.subheader("Generated Summary")
        st.write(summary)
        
        # Display statistics
        st.subheader("Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Input Length", f"{len(text_input.split())} words")
        with col2:
            st.metric("Summary Length", f"{len(summary.split())} words")