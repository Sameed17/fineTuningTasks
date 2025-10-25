"""
Streamlit App for BERT Sentiment Analysis
Real-time customer feedback sentiment classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .positive { color: #28a745; font-weight: bold; }
    .negative { color: #dc3545; font-weight: bold; }
    .neutral { color: #ffc107; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    try:
        model_path = "models/sentiment_bert"
        
        if not Path(model_path).exists():
            st.error("❌ Model not found! Please train the model first by running: python bert_sentiment_analysis.py")
            return None, None
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for a single text"""
    # Tokenize input
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
    
    # Get results
    confidence = probabilities[0][predicted_class].item()
    class_names = ['Negative', 'Neutral', 'Positive']
    
    class_probabilities = {
        class_names[i]: probabilities[0][i].item() 
        for i in range(len(class_names))
    }
    
    return {
        'predicted_class': class_names[predicted_class],
        'confidence': confidence,
        'class_probabilities': class_probabilities
    }

def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment"""
    emojis = {
        'Positive': '😊',
        'Negative': '😞',
        'Neutral': '😐'
    }
    return emojis.get(sentiment, '❓')

def get_sentiment_color(sentiment):
    """Get color for sentiment"""
    colors = {
        'Positive': '#28a745',
        'Negative': '#dc3545',
        'Neutral': '#ffc107'
    }
    return colors.get(sentiment, '#666')

# Main UI
st.markdown('<div class="main-header">😊 Sentiment Analysis App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time Customer Feedback Classification using BERT</div>', unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    model, tokenizer = load_model()

if model and tokenizer:
    st.session_state.model_loaded = True
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio("Choose a page", ["Single Prediction", "Batch Prediction", "History", "About"])
    
    # Single Prediction Page
    if page == "Single Prediction":
        st.header("📝 Single Text Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter your text here:",
                height=150,
                placeholder="Example: I absolutely love this product! It's amazing and works perfectly."
            )
            
            if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
                if text_input.strip():
                    # Make prediction
                    result = predict_sentiment(text_input, model, tokenizer)
                    
                    # Display result
                    st.markdown("### 🎯 Prediction Result")
                    
                    col3, col4, col5 = st.columns(3)
                    
                    with col3:
                        st.metric(
                            "Predicted Sentiment",
                            f"{get_sentiment_emoji(result['predicted_class'])} {result['predicted_class']}"
                        )
                    
                    with col4:
                        st.metric(
                            "Confidence",
                            f"{result['confidence']:.1%}"
                        )
                    
                    with col5:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.caption(f"Analyzed at: {timestamp}")
                    
                    # Sentiment bar
                    sentiment = result['predicted_class']
                    sentiment_color = get_sentiment_color(sentiment)
                    
                    st.markdown(f"""
                    <div style='background: {sentiment_color}; color: white; padding: 1rem; border-radius: 5px; text-align: center; font-size: 1.5rem; font-weight: bold;'>
                        {sentiment} Sentiment Detected
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.markdown("### 📊 Probability Distribution")
                    
                    probs = result['class_probabilities']
                    df_probs = pd.DataFrame({
                        'Sentiment': list(probs.keys()),
                        'Probability': list(probs.values())
                    })
                    
                    # Create bar chart
                    fig = px.bar(
                        df_probs,
                        x='Sentiment',
                        y='Probability',
                        color='Sentiment',
                        color_discrete_map={
                            'Positive': '#28a745',
                            'Negative': '#dc3545',
                            'Neutral': '#ffc107'
                        },
                        text='Probability',
                        labels={'Probability': 'Probability', 'Sentiment': 'Sentiment'}
                    )
                    fig.update_traces(
                        texttemplate='%{text:.1%}',
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=400,
                        yaxis=dict(
                            tickformat='.0%',
                            title="Probability"
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save to history
                    st.session_state.predictions_history.append({
                        'text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                        'sentiment': result['predicted_class'],
                        'confidence': result['confidence'],
                        'timestamp': timestamp
                    })
                else:
                    st.warning("⚠️ Please enter some text to analyze.")
        
        with col2:
            st.markdown("### 💡 Quick Examples")
            
            examples = [
                "I love this product!",
                "Terrible quality, very disappointed.",
                "It's okay, nothing special.",
                "Absolutely amazing experience!",
                "Worst purchase ever."
            ]
            
            for example in examples:
                if st.button(f"📄 {example}", use_container_width=True, key=example):
                    st.session_state.example_text = example
                    st.rerun()
            
            if 'example_text' in st.session_state:
                # Trigger prediction
                result = predict_sentiment(st.session_state.example_text, model, tokenizer)
                st.json({
                    'text': st.session_state.example_text,
                    'sentiment': result['predicted_class'],
                    'confidence': f"{result['confidence']:.1%}"
                })
    
    # Batch Prediction Page
    elif page == "Batch Prediction":
        st.header("📊 Batch Prediction")
        
        st.markdown("Upload a CSV file or enter multiple texts to analyze sentiment in bulk.")
        
        tab1, tab2 = st.tabs(["CSV Upload", "Manual Entry"])
        
        with tab1:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df.head())
                    
                    # Select text column
                    text_column = st.selectbox("Select the column containing text", df.columns)
                    
                    if st.button("🚀 Process All", type="primary"):
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, text in enumerate(df[text_column]):
                            result = predict_sentiment(str(text), model, tokenizer)
                            results.append({
                                'text': text,
                                'prediction': result['predicted_class'],
                                'confidence': f"{result['confidence']:.1%}"
                            })
                            progress_bar.progress((i + 1) / len(df))
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        
        with tab2:
            manual_texts = st.text_area(
                "Enter texts (one per line):",
                height=200,
                placeholder="I love this product!\nTerrible quality...\nIt's okay."
            )
            
            if st.button("🔍 Analyze All", type="primary"):
                texts = [line.strip() for line in manual_texts.split('\n') if line.strip()]
                
                if texts:
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, text in enumerate(texts):
                        result = predict_sentiment(text, model, tokenizer)
                        results.append({
                            'text': text,
                            'sentiment': get_sentiment_emoji(result['predicted_class']) + " " + result['predicted_class'],
                            'confidence': f"{result['confidence']:.1%}"
                        })
                        progress_bar.progress((i + 1) / len(texts))
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
    
    # History Page
    elif page == "History":
        st.header("📜 Prediction History")
        
        if st.session_state.predictions_history:
            df_history = pd.DataFrame(st.session_state.predictions_history)
            st.dataframe(df_history, use_container_width=True)
            
            # Statistics
            st.markdown("### 📈 Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", len(df_history))
            
            with col2:
                positive_count = len(df_history[df_history['sentiment'] == 'Positive'])
                st.metric("Positive", positive_count)
            
            with col3:
                negative_count = len(df_history[df_history['sentiment'] == 'Negative'])
                st.metric("Negative", negative_count)
            
            with col4:
                neutral_count = len(df_history[df_history['sentiment'] == 'Neutral'])
                st.metric("Neutral", neutral_count)
            
            # Pie chart
            sentiment_counts = df_history['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={
                    'Positive': '#28a745',
                    'Negative': '#dc3545',
                    'Neutral': '#ffc107'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.predictions_history = []
                st.rerun()
        else:
            st.info("No predictions yet. Start analyzing texts to see history here.")
    
    # About Page
    else:
        st.header("📖 About")
        
        st.markdown("""
        ### What is this app?
        This is a real-time sentiment analysis application powered by BERT (Bidirectional Encoder Representations from Transformers).
        
        ### Features:
        - ✅ **Single Text Prediction**: Analyze individual customer feedback
        - ✅ **Batch Processing**: Upload CSV files or enter multiple texts
        - ✅ **Prediction History**: Track all your predictions
        - ✅ **Visual Analytics**: Interactive charts and statistics
        - ✅ **High Accuracy**: Fine-tuned BERT model for customer feedback
        
        ### Model Information:
        - **Base Model**: BERT-base-uncased
        - **Training Dataset**: Customer feedback dataset with 3 sentiment classes
        - **Classes**: Positive, Negative, Neutral
        - **Sequence Length**: 128 tokens
        
        ### How to use:
        1. Navigate to **Single Prediction** for individual texts
        2. Use **Batch Prediction** for multiple texts or CSV files
        3. Check **History** to view all your predictions
        4. Download results for further analysis
        
        ### Technical Stack:
        - **Framework**: Streamlit
        - **Model**: Hugging Face Transformers
        - **Backend**: PyTorch
        - **Visualization**: Plotly
        
        ### Development:
        Developed as part of NLP Assignment 3 - Task 1: Encoder-Only (BERT) Customer Feedback Classification
        """)
else:
    st.error("""
    ## ⚠️ Model Not Found
    
    Please train the model first by running:
    
    ```bash
    python bert_sentiment_analysis.py
    ```
    
    This will train the BERT model and save it to the `models/sentiment_bert` directory.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Made with ❤️ using Streamlit and BERT | "
    "NLP Assignment 3 - Sentiment Analysis"
    "</div>",
    unsafe_allow_html=True
)
