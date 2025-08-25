# Fake News Detection Project

This project is a machine learning-based application for detecting fake news. It leverages natural language processing and classification algorithms to distinguish between real and fake news articles.

## Features

- Trained machine learning model for fake news detection (`fake_news_model.pkl`)
- Dataset of real and fake news articles (`fake_or_real_news.csv`)
- Visualization of model performance (confusion matrix, word clouds)
- Python scripts for data processing, model training, and prediction (`main.py`, `app.py`)

## Project Structure

- `main.py`: Main script for training and evaluating the model.
- `app.py`: Application script (could be a web app or CLI for predictions).
- `fake_news_model.pkl`: Pre-trained machine learning model.
- `fake_or_real_news.csv`: Dataset used for training/testing.
- `confusion_matrix.png`: Visualization of model performance.
- `wordclouds.png`: Word clouds for data exploration.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Anique-1/DEN-Intership.git
   cd DEN-Intership
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is not present, install common packages: pandas, scikit-learn, matplotlib, wordcloud, etc.)*

## Usage

- To train or evaluate the model, run:
  ```bash
  python main.py
  ```

- To use the application (e.g., for predictions), run:
  ```bash
  python app.py
  ```

## Visualizations

- `confusion_matrix.png`: Shows the performance of the classifier.
- `wordclouds.png`: Displays the most frequent words in real and fake news.

## Dataset

- The dataset `fake_or_real_news.csv` contains labeled news articles for training and testing the model.

## License

This project is for educational purposes.

## Contact

For questions or contributions, please visit the [GitHub repository](https://github.com/Anique-1/DEN-Intership).
