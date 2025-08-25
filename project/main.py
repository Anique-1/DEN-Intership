import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.pipeline = None
    
    def preprocess_text(self, text):
        """
        Preprocess text data by removing stopwords, punctuation, numbers and applying lemmatization
        """
        if pd.isna(text):
            return ""
        
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
            if token not in self.stop_words and len(token) > 2:
                lemmatized_token = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized_token)
        
        return ' '.join(processed_tokens)
    
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the dataset
        """
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        # Check for missing values
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # Handle missing values
        df = df.dropna(subset=['title', 'text', 'label'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title', 'text'])
        print(f"Dataset shape after cleaning: {df.shape}")
        
        # Combine title and text for better feature extraction
        df['combined_text'] = df['title'] + ' ' + df['text']
        
        # Preprocess text
        print("Preprocessing text...")
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Map labels to binary (FAKE=0, REAL=1)
        df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
        
        return df
    
    def train_model(self, df):
        """
        Train the logistic regression model
        """
        print("Training model...")
        
        X = df['processed_text']
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Print classification report
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'])}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)
        
        # Save model
        self.save_model()
        
        return X_test, y_test, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['FAKE', 'REAL'], 
                   yticklabels=['FAKE', 'REAL'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_wordclouds(self, df):
        """
        Generate word clouds for fake and real news
        """
        fake_text = ' '.join(df[df['label'] == 0]['processed_text'].values)
        real_text = ' '.join(df[df['label'] == 1]['processed_text'].values)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Fake news word cloud
        fake_wordcloud = WordCloud(width=400, height=300, background_color='white').generate(fake_text)
        axes[0].imshow(fake_wordcloud, interpolation='bilinear')
        axes[0].set_title('Fake News Word Cloud')
        axes[0].axis('off')
        
        # Real news word cloud
        real_wordcloud = WordCloud(width=400, height=300, background_color='white').generate(real_text)
        axes[1].imshow(real_wordcloud, interpolation='bilinear')
        axes[1].set_title('Real News Word Cloud')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_text(self, text):
        """
        Predict if a single text is fake or real
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")
        
        processed_text = self.preprocess_text(text)
        prediction = self.pipeline.predict([processed_text])[0]
        probability = self.pipeline.predict_proba([processed_text])[0]
        
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
    
    def save_model(self):
        """
        Save the trained model
        """
        with open('fake_news_model.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)
        print("Model saved as 'fake_news_model.pkl'")
    
    def load_model(self):
        """
        Load a pre-trained model
        """
        try:
            with open('fake_news_model.pkl', 'rb') as f:
                self.pipeline = pickle.load(f)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("No saved model found. Please train the model first.")

def main():
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Load and preprocess data
    df = detector.load_and_preprocess_data('fake_or_real_news.csv')
    
    # Train model
    X_test, y_test, y_pred, y_pred_proba = detector.train_model(df)
    
    # Generate visualizations
    detector.generate_wordclouds(df)
    
    # Test with sample predictions
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    # Sample fake news text
    fake_sample = "BREAKING: Scientists discover that vaccines cause autism in new shocking study!"
    result = detector.predict_single_text(fake_sample)
    print(f"\nSample Text: {fake_sample}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: FAKE={result['probabilities']['FAKE']:.4f}, REAL={result['probabilities']['REAL']:.4f}")
    
    # Sample real news text
    real_sample = "The Federal Reserve announced today that it will maintain interest rates at current levels following the monthly policy meeting."
    result = detector.predict_single_text(real_sample)
    print(f"\nSample Text: {real_sample}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: FAKE={result['probabilities']['FAKE']:.4f}, REAL={result['probabilities']['REAL']:.4f}")

if __name__ == "__main__":
    main()