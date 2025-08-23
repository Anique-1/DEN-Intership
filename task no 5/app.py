import streamlit as st
import joblib
import os

# Set page config for a stylish look
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for stylish design
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 1.5px solid #6366f1;
        padding: 0.5rem 1rem;
    }
    .stButton>button {
        background: #6366f1;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: #4338ca;
    }
    .result-box {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 1.2rem;
        font-weight: 500;
        color: #334155;
        box-shadow: 0 2px 8px rgba(99,102,241,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and description
st.markdown(
    "<div class='main'>"
    "<h1 style='color:#6366f1;'>💬 Sentiment Analysis</h1>"
    "<p style='color:#334155;'>Enter a review or sentence below to analyze its sentiment using our AI model.</p>",
    unsafe_allow_html=True
)

# Load the trained model and vectorizer
MODEL_PATH = os.path.join("saved_models", "best_sentiment_classifier_naive_bayes_tf-idf_20250823_174825.pkl")
@st.cache_resource
def load_model_and_vectorizer():
    obj = joblib.load(MODEL_PATH)
    model = obj["model"]
    vectorizer = obj["tfidf_vectorizer"]
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Text input
user_input = st.text_area("Enter your review or sentence:", height=100, key="input_text")

# Predict button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform input and predict
        try:
            X = vectorizer.transform([user_input])
            prediction = model.predict(X)[0]
            # Optionally, map prediction to a more user-friendly label
            if prediction == 1 or prediction == "positive":
                sentiment = "😊 Positive"
                color = "#22c55e"
            elif prediction == 0 or prediction == "neutral":
                sentiment = "😐 Neutral"
                color = "#facc15"
            else:
                sentiment = "😞 Negative"
                color = "#ef4444"
            st.markdown(
                f"<div class='result-box' style='border-left: 6px solid {color};'>"
                f"Sentiment: <span style='color:{color}; font-weight:bold;'>{sentiment}</span>"
                "</div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("</div>", unsafe_allow_html=True)
