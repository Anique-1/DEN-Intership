import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import pickle
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

class ReviewSentimentClassifier:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {}
        self.results = {}
        self.best_model_info = None
        self.model_save_dir = "saved_models"

    def load_and_inspect_data(self, file_path):
        """Load and inspect the multilingual mobile app review dataset"""
        print("=" * 50)
        print("STEP 1: DATA LOADING AND INSPECTION")
        print("=" * 50)

        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        print("\nSample rows:")
        print(df.head(3).to_string())

        # Check if expected columns exist
        expected_columns = ['review_text', 'rating']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"\nWarning: Missing expected columns: {missing_columns}")
            print("Available columns:", df.columns.tolist())
            # Try to find similar column names
            for missing_col in missing_columns:
                similar_cols = [col for col in df.columns if missing_col.lower() in col.lower() or col.lower() in missing_col.lower()]
                if similar_cols:
                    print(f"Possible alternatives for '{missing_col}': {similar_cols}")

        # Display distributions for available columns
        if 'review_language' in df.columns:
            print("\nReview language distribution:")
            print(df['review_language'].value_counts())

        if 'rating' in df.columns:
            print("\nRating distribution:")
            print(df['rating'].value_counts().sort_index())

        if 'verified_purchase' in df.columns:
            print("\nVerified purchase distribution:")
            print(df['verified_purchase'].value_counts())

        if 'device_type' in df.columns:
            print("\nDevice type distribution:")
            print(df['device_type'].value_counts())

        if 'review_text' in df.columns:
            print("\nSample review texts:")
            for i, row in df.head(3).iterrows():
                print(f"{i+1}. {str(row['review_text'])[:100]}...")
        
        return df
    
    def clean_text(self, text):
        """Clean and preprocess review text"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def prepare_data(self, df):
        """Clean and prepare the review dataset for sentiment classification"""
        print("\n" + "=" * 50)
        print("STEP 2: DATA PREPARATION")
        print("=" * 50)

        # Check required columns
        if 'review_text' not in df.columns:
            raise ValueError("'review_text' column not found in dataset")
        if 'rating' not in df.columns:
            raise ValueError("'rating' column not found in dataset")

        # Remove rows with missing review_text or rating
        initial_rows = len(df)
        df = df.dropna(subset=['review_text', 'rating'])
        print(f"Removed {initial_rows - len(df)} rows with missing data")

        # Create sentiment label from rating
        def get_sentiment(rating):
            try:
                r = float(rating)
                if r >= 4.0:
                    return "positive"
                elif r <= 2.0:
                    return "negative"
                else:
                    return "neutral"
            except:
                return "neutral"

        df['sentiment'] = df['rating'].apply(get_sentiment)
        print("Sentiment label distribution:")
        print(df['sentiment'].value_counts())

        print("Cleaning review text...")
        df['cleaned_review'] = df['review_text'].apply(self.clean_text)
        
        # Remove empty cleaned reviews
        df = df[df['cleaned_review'].str.len() > 0]
        print(f"Final dataset size after cleaning: {len(df)}")

        # Encode sentiment labels
        df['sentiment_encoded'] = self.label_encoder.fit_transform(df['sentiment'])

        # Split into train and test sets (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_review'], df['sentiment_encoded'],
            test_size=0.2, random_state=42, stratify=df['sentiment_encoded']
        )

        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print("Training set sentiment distribution:")
        print(pd.Series(y_train).value_counts())

        return X_train, X_test, y_train, y_test
    
    def extract_tfidf_features(self, X_train, X_test):
        """Extract TF-IDF features"""
        print("\nExtracting TF-IDF features...")
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
        print(f"Number of features: {X_train_tfidf.shape[1]}")
        
        return X_train_tfidf, X_test_tfidf
    
    def extract_word2vec_features(self, X_train, X_test):
        """Extract Word2Vec features"""
        print("\nExtracting Word2Vec features...")
        
        # Prepare sentences for Word2Vec
        train_sentences = [text.split() for text in X_train]
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=train_sentences,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            sg=1  # Skip-gram
        )
        
        def get_document_vector(text):
            """Get average word vector for a document"""
            words = text.split()
            word_vectors = []
            for word in words:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            
            if word_vectors:
                return np.mean(word_vectors, axis=0)
            else:
                return np.zeros(100)  # Return zero vector if no words found
        
        # Convert documents to vectors
        X_train_w2v = np.array([get_document_vector(text) for text in X_train])
        X_test_w2v = np.array([get_document_vector(text) for text in X_test])
        
        print(f"Word2Vec feature matrix shape: {X_train_w2v.shape}")
        print(f"Vector dimension: {X_train_w2v.shape[1]}")
        
        return X_train_w2v, X_test_w2v
    
    def train_models(self, X_train_tfidf, X_test_tfidf, X_train_w2v, X_test_w2v, y_train, y_test):
        """Train different models on both feature sets"""
        print("\n" + "=" * 50)
        print("STEP 4: MODEL TRAINING")
        print("=" * 50)
        
        # Initialize models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'naive_bayes': MultinomialNB()
        }
        
        feature_sets = {
            'TF-IDF': (X_train_tfidf, X_test_tfidf),
            'Word2Vec': (X_train_w2v, X_test_w2v)
        }
        
        results = {}
        
        for model_name, model in models.items():
            results[model_name] = {}
            
            for feature_name, (X_train_feat, X_test_feat) in feature_sets.items():
                print(f"\nTraining {model_name} with {feature_name} features...")
                
                # Skip Naive Bayes for Word2Vec (requires non-negative features)
                if model_name == 'naive_bayes' and feature_name == 'Word2Vec':
                    print("Skipping Naive Bayes with Word2Vec (requires non-negative features)")
                    continue
                
                # Train model
                model.fit(X_train_feat, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_feat)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results[model_name][feature_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'y_pred': y_pred,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"F1-Score: {f1:.4f}")
        
        self.results = results
        return results
    
    def evaluate_models(self, y_test):
        """Evaluate and compare all models"""
        print("\n" + "=" * 50)
        print("STEP 5: MODEL EVALUATION AND COMPARISON")
        print("=" * 50)
        
        # Create comparison dataframe
        comparison_data = []
        
        for model_name, model_results in self.results.items():
            for feature_name, metrics in model_results.items():
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Features': feature_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\nModel Performance Comparison:")
        print("=" * 80)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Find best model
        best_idx = comparison_df['F1-Score'].idxmax()
        best_model_info = comparison_df.iloc[best_idx]
        
        print(f"\nBest Model: {best_model_info['Model']} with {best_model_info['Features']} features")
        print(f"Best F1-Score: {best_model_info['F1-Score']:.4f}")
        
        # Store best model info for saving
        self.best_model_info = {
            'model_name': best_model_info['Model'].lower().replace(' ', '_'),
            'feature_type': best_model_info['Features'],
            'f1_score': best_model_info['F1-Score'],
            'metrics': best_model_info
        }
        
        return comparison_df
    
    def plot_results(self, comparison_df, y_test):
        """Plot evaluation results"""
        print("\nGenerating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Performance Comparison
        plt.subplot(2, 3, 1)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(comparison_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, comparison_df[metric], width, 
                   label=metric, alpha=0.8)
        
        plt.xlabel('Model-Feature Combinations')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.legend()
        plt.xticks(x + width*1.5, 
                  [f"{row['Model']}\n({row['Features']})" for _, row in comparison_df.iterrows()], 
                  rotation=45)
        
        # 2. F1-Score Comparison
        plt.subplot(2, 3, 2)
        colors = plt.cm.viridis(np.linspace(0, 1, len(comparison_df)))
        bars = plt.bar(range(len(comparison_df)), comparison_df['F1-Score'], color=colors)
        plt.xlabel('Model-Feature Combinations')
        plt.ylabel('F1-Score')
        plt.title('F1-Score Comparison')
        plt.xticks(range(len(comparison_df)), 
                  [f"{row['Model']}\n({row['Features']})" for _, row in comparison_df.iterrows()], 
                  rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, comparison_df['F1-Score']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Feature comparison
        plt.subplot(2, 3, 3)
        feature_performance = comparison_df.groupby('Features')['F1-Score'].mean()
        plt.bar(feature_performance.index, feature_performance.values, color=['skyblue', 'lightcoral'])
        plt.xlabel('Feature Type')
        plt.ylabel('Average F1-Score')
        plt.title('Feature Type Performance')
        
        # 4. Model comparison (average across features)
        plt.subplot(2, 3, 4)
        model_performance = comparison_df.groupby('Model')['F1-Score'].mean()
        plt.bar(model_performance.index, model_performance.values, color=['lightgreen', 'gold', 'salmon'])
        plt.xlabel('Model Type')
        plt.ylabel('Average F1-Score')
        plt.title('Model Type Performance')
        plt.xticks(rotation=45)
        
        # 5. Best model confusion matrix
        best_idx = comparison_df['F1-Score'].idxmax()
        best_model_name = comparison_df.iloc[best_idx]['Model'].lower().replace(' ', '_')
        best_feature_name = comparison_df.iloc[best_idx]['Features']
        
        best_cm = self.results[best_model_name][best_feature_name]['confusion_matrix']
        
        plt.subplot(2, 3, 5)
        # Get actual class names for confusion matrix
        class_names = self.label_encoder.classes_
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Best Model\n({comparison_df.iloc[best_idx]["Model"]} with {best_feature_name})')
        
        # 6. Metrics radar chart for best model
        plt.subplot(2, 3, 6, projection='polar')
        best_metrics = comparison_df.iloc[best_idx][['Accuracy', 'Precision', 'Recall', 'F1-Score']].values
        angles = np.linspace(0, 2*np.pi, len(best_metrics), endpoint=False).tolist()
        best_metrics = best_metrics.tolist()
        
        # Close the plot
        angles += angles[:1]
        best_metrics += best_metrics[:1]
        
        plt.plot(angles, best_metrics, 'o-', linewidth=2, label='Best Model')
        plt.fill(angles, best_metrics, alpha=0.25)
        plt.xticks(angles[:-1], ['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        plt.ylim(0, 1)
        plt.title(f'Best Model Metrics\n({comparison_df.iloc[best_idx]["Model"]} with {best_feature_name})')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def detailed_classification_report(self, y_test):
        """Print detailed classification reports for all models"""
        print("\n" + "=" * 50)
        print("DETAILED CLASSIFICATION REPORTS")
        print("=" * 50)

        # Get sentiment classes from the label encoder
        sentiment_classes = list(self.label_encoder.classes_)

        for model_name, model_results in self.results.items():
            for feature_name, metrics in model_results.items():
                print(f"\n{model_name.replace('_', ' ').title()} with {feature_name} Features:")
                print("-" * 60)
                y_pred = metrics['y_pred']
                print(classification_report(y_test, y_pred,
                                           target_names=sentiment_classes))
    
    def save_best_model(self):
        """Save the best performing model and all necessary components"""
        if self.best_model_info is None:
            print("Error: No best model information available. Run complete analysis first.")
            return None
        
        print("\n" + "=" * 50)
        print("SAVING BEST MODEL")
        print("=" * 50)
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.best_model_info['model_name']
        feature_type = self.best_model_info['feature_type']
        
        # Get the best model
        best_model = self.results[model_name][feature_type]['model']
        
        # Prepare save paths
        model_filename = f"best_sentiment_classifier_{model_name}_{feature_type.lower()}_{timestamp}.pkl"
        model_path = os.path.join(self.model_save_dir, model_filename)

        # Prepare the complete model package
        model_package = {
            'model': best_model,
            'model_name': model_name,
            'feature_type': feature_type,
            'label_encoder': self.label_encoder,
            'tfidf_vectorizer': self.tfidf_vectorizer if feature_type == 'TF-IDF' else None,
            'word2vec_model': self.word2vec_model if feature_type == 'Word2Vec' else None,
            'stop_words': self.stop_words,
            'best_model_info': self.best_model_info,
            'timestamp': timestamp,
            'performance_metrics': {
                'accuracy': self.results[model_name][feature_type]['accuracy'],
                'precision': self.results[model_name][feature_type]['precision'],
                'recall': self.results[model_name][feature_type]['recall'],
                'f1_score': self.results[model_name][feature_type]['f1_score']
            }
        }

        # Save the complete model package
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_package, f)

            print(f"✅ Best model saved successfully!")
            print(f"📁 File path: {model_path}")
            print(f"🎯 Model: {model_name.replace('_', ' ').title()}")
            print(f"🔧 Features: {feature_type}")
            print(f"📊 F1-Score: {self.best_model_info['f1_score']:.4f}")
            print(f"⏰ Timestamp: {timestamp}")

            # Also save a summary file
            summary_path = os.path.join(self.model_save_dir, f"model_summary_{timestamp}.txt")
            with open(summary_path, 'w') as f:
                f.write("SENTIMENT CLASSIFIER MODEL SUMMARY\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Model Name: {model_name.replace('_', ' ').title()}\n")
                f.write(f"Feature Type: {feature_type}\n")
                f.write(f"Save Timestamp: {timestamp}\n")
                f.write(f"Model File: {model_filename}\n\n")
                f.write("PERFORMANCE METRICS:\n")
                f.write("-" * 20 + "\n")
                for metric, value in model_package['performance_metrics'].items():
                    f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")

            print(f"📄 Model summary saved: {summary_path}")

            return model_path

        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return None
    
    def load_saved_model(self, model_path):
        """Load a previously saved model"""
        try:
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            print("✅ Model loaded successfully!")
            print(f"🎯 Model: {model_package['model_name'].replace('_', ' ').title()}")
            print(f"🔧 Features: {model_package['feature_type']}")
            print(f"📊 F1-Score: {model_package['performance_metrics']['f1_score']:.4f}")
            print(f"⏰ Saved on: {model_package['timestamp']}")
            
            return model_package
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return None
    
    def predict_with_saved_model(self, model_package, messages):
        """Make predictions using a saved model"""
        if model_package is None:
            print("Error: No model package provided.")
            return None
        
        try:
            # Clean the input messages
            if isinstance(messages, str):
                messages = [messages]
            
            cleaned_messages = [self.clean_text(msg) for msg in messages]
            
            # Extract features based on the model's feature type
            feature_type = model_package['feature_type']
            model = model_package['model']
            
            if feature_type == 'TF-IDF':
                tfidf_vectorizer = model_package['tfidf_vectorizer']
                features = tfidf_vectorizer.transform(cleaned_messages)
            
            elif feature_type == 'Word2Vec':
                word2vec_model = model_package['word2vec_model']
                
                def get_document_vector(text):
                    words = text.split()
                    word_vectors = []
                    for word in words:
                        if word in word2vec_model.wv:
                            word_vectors.append(word2vec_model.wv[word])
                    
                    if word_vectors:
                        return np.mean(word_vectors, axis=0)
                    else:
                        return np.zeros(100)
                
                features = np.array([get_document_vector(text) for text in cleaned_messages])
            
            # Make predictions
            predictions = model.predict(features)
            prediction_proba = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
            
            # Convert predictions back to labels
            label_encoder = model_package['label_encoder']
            predicted_labels = label_encoder.inverse_transform(predictions)
            
            # Prepare results
            results = []
            for i, (original_msg, predicted_label) in enumerate(zip(messages, predicted_labels)):
                result = {
                    'message': original_msg,
                    'prediction': predicted_label,
                    'confidence': prediction_proba[i].max() if prediction_proba is not None else None
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error making predictions: {str(e)}")
            return None
    
    def demonstrate_saved_model(self, model_path, test_messages=None):
        """Demonstrate the saved model with sample messages"""
        print("\n" + "=" * 50)
        print("DEMONSTRATING SAVED MODEL")
        print("=" * 50)
        
        # Load the model
        model_package = self.load_saved_model(model_path)
        if model_package is None:
            return
        
        # Use default test messages if none provided
        if test_messages is None:
            test_messages = [
                "This app is absolutely amazing! I love using it every day!",
                "Terrible app, keeps crashing and wasting my time.",
                "It's okay, has some good features but room for improvement.",
                "Best app ever! Highly recommended to everyone!",
                "Not bad but could be better with more features."
            ]
        
        print("\nMaking predictions on test messages:")
        print("-" * 40)
        
        # Make predictions
        results = self.predict_with_saved_model(model_package, test_messages)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Message: \"{result['message'][:80]}{'...' if len(result['message']) > 80 else ''}\"")
                print(f"   Prediction: {result['prediction'].upper()}")
                if result['confidence']:
                    print(f"   Confidence: {result['confidence']:.3f}")
        
        return results
    
    def run_complete_analysis(self, file_path):
        """Run the complete sentiment classification analysis"""
        # Step 1: Load and inspect data
        df = self.load_and_inspect_data(file_path)
        
        # Step 2: Prepare data (FIXED: now returns only 4 values)
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Step 3: Feature extraction
        print("\n" + "=" * 50)
        print("STEP 3: FEATURE EXTRACTION")
        print("=" * 50)
        
        X_train_tfidf, X_test_tfidf = self.extract_tfidf_features(X_train, X_test)
        X_train_w2v, X_test_w2v = self.extract_word2vec_features(X_train, X_test)
        
        # Step 4: Train models
        results = self.train_models(X_train_tfidf, X_test_tfidf, 
                                  X_train_w2v, X_test_w2v, y_train, y_test)
        
        # Step 5: Evaluate models
        comparison_df = self.evaluate_models(y_test)
        
        # Generate detailed reports
        self.detailed_classification_report(y_test)
        
        # Plot results
        self.plot_results(comparison_df, y_test)
        
        # Save the best model
        model_path = self.save_best_model()
        
        # Demonstrate the saved model
        if model_path:
            self.demonstrate_saved_model(model_path)
        
        return comparison_df, model_path

# Usage example
if __name__ == "__main__":
    # Initialize the classifier
    classifier = ReviewSentimentClassifier()

    # Run the complete analysis
    try:
        results_df, model_path = classifier.run_complete_analysis('multilingual_mobile_app_reviews_2025.csv')
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nKey Findings:")
        print("1. Dataset contains multilingual mobile app reviews with ratings and metadata.")
        print("2. Sentiment labels were derived from ratings (positive/neutral/negative).")
        print("3. Text features (TF-IDF, Word2Vec) and classifiers were compared.")
        print("4. Performance metrics and visualizations provide comprehensive evaluation.")
        print("5. Best performing model has been saved for future use.")

        # Example of loading and using the saved model later
        print("\n" + "=" * 50)
        print("EXAMPLE: USING SAVED MODEL FOR NEW PREDICTIONS")
        print("=" * 50)

        if model_path:
            saved_model = classifier.load_saved_model(model_path)
            new_reviews = [
                "This app is amazing! I use it every day.",
                "Terrible update, keeps crashing on my phone.",
                "Good features but too many ads.",
                "Not bad, but could be improved.",
            ]
            predictions = classifier.predict_with_saved_model(saved_model, new_reviews)
            if predictions:
                print("\nPredictions for new reviews:")
                for pred in predictions:
                    sentiment_prob = "HIGH" if pred['confidence'] and pred['confidence'] > 0.7 else "MEDIUM" if pred['confidence'] and pred['confidence'] > 0.5 else "LOW"
                    print(f"'{pred['message'][:50]}...' -> {pred['prediction']} (Confidence: {sentiment_prob})")

    except FileNotFoundError:
        print("Error: multilingual_mobile_app_reviews_2025.csv file not found. Please ensure the file is in the correct path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data file and try again.")

# Additional utility functions for model management
def list_saved_models(models_dir="saved_models"):
    """List all saved models in the directory"""
    if not os.path.exists(models_dir):
        print(f"No saved models directory found: {models_dir}")
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("No saved models found.")
        return []
    
    print(f"\nSaved Models in '{models_dir}':")
    print("-" * 40)
    
    for i, filename in enumerate(model_files, 1):
        # Extract info from filename
        parts = filename.replace('.pkl', '').split('_')
        if len(parts) >= 6:
            model_type = parts[3].replace('-', ' ').title()
            feature_type = parts[4].replace('-', ' ')
            timestamp = f"{parts[5]}_{parts[6]}" if len(parts) > 6 else parts[5]
            print(f"{i}. {filename}")
            print(f"   Model: {model_type}")
            print(f"   Features: {feature_type}")
            print(f"   Date: {timestamp[:8]} {timestamp[9:11]}:{timestamp[11:13]}")
        else:
            print(f"{i}. {filename}")
    
    return model_files

def load_and_predict(model_path, messages):
    """Convenience function to load a model and make predictions"""
    classifier = ReviewSentimentClassifier()
    model_package = classifier.load_saved_model(model_path)

    if model_package:
        return classifier.predict_with_saved_model(model_package, messages)
    return None