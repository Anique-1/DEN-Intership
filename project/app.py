import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px

# Configure Streamlit page
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data if not already present
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('fake_news_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

# Text preprocessing function
@st.cache_data
def preprocess_text(text):
    """Preprocess text for prediction"""
    if pd.isna(text) or text == "":
        return ""
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and apply lemmatization
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            lemmatized_token = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized_token)
    
    return ' '.join(processed_tokens)

# Prediction function
def predict_news(text, model):
    """Make prediction on input text"""
    if model is None:
        return None
    
    processed_text = preprocess_text(text)
    if processed_text == "":
        return None
    
    prediction = model.predict([processed_text])[0]
    probability = model.predict_proba([processed_text])[0]
    
    label = "REAL" if prediction == 1 else "FAKE"
    confidence = max(probability)
    
    return {
        'prediction': label,
        'confidence': confidence,
        'probabilities': {
            'FAKE': probability[0],
            'REAL': probability[1]
        }
    }

# Generate word cloud
@st.cache_data
def generate_wordcloud(text):
    """Generate word cloud for given text"""
    if text and len(text.strip()) > 0:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis'
        ).generate(text)
        return wordcloud
    return None

def main():
    # Download NLTK data
    download_nltk_data()
    
    # Load model
    model = load_model()
    
    # Title and description
    st.title("🔍 Fake News Detector")
    st.markdown("""
    This application uses Machine Learning to detect whether a news article is **FAKE** or **REAL**.
    The model is trained using TF-IDF features and Logistic Regression classifier.
    """)
    
    # Sidebar
    st.sidebar.header("📊 Model Information")
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.info("""
        **Model Details:**
        - Algorithm: Logistic Regression
        - Features: TF-IDF Vectorization
        - Preprocessing: Stopword removal, Lemmatization
        - Training Data: Fake and Real News Dataset
        """)
    else:
        st.sidebar.error("❌ Model not found! Please run main.py first to train the model.")
        st.error("**Model not found!** Please run `main.py` first to train the model, then restart this app.")
        return
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter News Article")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload File"])
        
        user_input = ""
        
        if input_method == "Type/Paste Text":
            user_input = st.text_area(
                "Paste your news article here:",
                height=200,
                placeholder="Enter the news article text you want to analyze..."
            )
        
        elif input_method == "Upload File":
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            if uploaded_file is not None:
                user_input = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", user_input, height=200)
        
        # Prediction button
        if st.button("🔍 Analyze Article", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing..."):
                    result = predict_news(user_input, model)
                
                if result:
                    # Display results
                    st.header("📊 Analysis Results")
                    
                    # Prediction result
                    if result['prediction'] == 'FAKE':
                        st.error(f"🚨 **FAKE NEWS** (Confidence: {result['confidence']:.2%})")
                    else:
                        st.success(f"✅ **REAL NEWS** (Confidence: {result['confidence']:.2%})")
                    
                    # Probability visualization
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['FAKE', 'REAL'],
                            y=[result['probabilities']['FAKE'], result['probabilities']['REAL']],
                            marker_color=['red' if result['prediction'] == 'FAKE' else 'lightcoral',
                                        'green' if result['prediction'] == 'REAL' else 'lightgreen']
                        )
                    ])
                    
                    fig.update_layout(
                        title="Prediction Probabilities",
                        xaxis_title="Classification",
                        yaxis_title="Probability",
                        yaxis=dict(range=[0, 1]),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed probabilities
                    col_fake, col_real = st.columns(2)
                    with col_fake:
                        st.metric("FAKE Probability", f"{result['probabilities']['FAKE']:.2%}")
                    with col_real:
                        st.metric("REAL Probability", f"{result['probabilities']['REAL']:.2%}")
                
                else:
                    st.error("Unable to analyze the text. Please check your input.")
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.header("📈 Text Analysis")
        
        if user_input.strip():
            # Basic text statistics
            word_count = len(user_input.split())
            char_count = len(user_input)
            
            st.metric("Word Count", word_count)
            st.metric("Character Count", char_count)
            
            # Generate word cloud
            processed_text = preprocess_text(user_input)
            if processed_text:
                st.subheader("☁️ Word Cloud")
                wordcloud = generate_wordcloud(processed_text)
                if wordcloud:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Not enough text to generate word cloud.")
        else:
            st.info("Enter text to see analysis statistics.")
    
    # Example articles section
    st.header("📋 Try These Example Articles")
    
    examples = {
        "Fake News Example": """
        BREAKING: Scientists Discover That Drinking Coffee Backwards Can Cure All Diseases!
        
        A groundbreaking study conducted by the imaginary University of Nowhere has revealed 
        that drinking coffee in reverse can cure everything from common cold to serious diseases. 
        The researchers claim that this method activates hidden healing properties in coffee beans. 
        Doctors are amazed by this discovery, though no peer-reviewed studies have been conducted.
        """,
        
        "Real News Example": """
        Federal Reserve Maintains Interest Rates Following Policy Meeting
        
        The Federal Reserve announced today that it will maintain the federal funds rate at current levels 
        following the conclusion of its two-day policy meeting. The decision was widely expected by economists 
        and market analysts who had anticipated no change in monetary policy at this time. Fed Chair Jerome Powell 
        cited current economic conditions and inflation metrics as key factors in the decision-making process.
        """
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fake News Example")
        if st.button("Try Fake Example"):
            st.session_state.example_text = examples["Fake News Example"]
    
    with col2:
        st.subheader("Real News Example")
        if st.button("Try Real Example"):
            st.session_state.example_text = examples["Real News Example"]
    
    # Display example if selected
    if 'example_text' in st.session_state:
        st.text_area("Example Article:", st.session_state.example_text, height=150)
        
        if st.button("🔍 Analyze Example", type="secondary"):
            with st.spinner("Analyzing example..."):
                result = predict_news(st.session_state.example_text, model)
            
            if result:
                if result['prediction'] == 'FAKE':
                    st.error(f"🚨 **FAKE NEWS** (Confidence: {result['confidence']:.2%})")
                else:
                    st.success(f"✅ **REAL NEWS** (Confidence: {result['confidence']:.2%})")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** This model is for educational purposes and may not be 100% accurate. 
    Always verify news from multiple reliable sources.
    
    **Developed by:** Digital Empowerment Network AI Project
    **Mentor:** Hussain Shoaib
    """)

if __name__ == "__main__":
    main()