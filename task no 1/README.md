# Student Performance Grade Prediction

A machine learning project that predicts student grade classes based on various academic and personal factors using Decision Tree and Random Forest algorithms.

## 📋 Project Overview

This project analyzes student performance data and builds predictive models to classify students into different grade categories. The system uses multiple features including academic metrics, personal characteristics, and extracurricular activities to predict student performance.

## 🎯 Features

- **Interactive Web Application**: Streamlit-based web interface for easy model interaction
- **Multiple ML Models**: Decision Tree and Random Forest classifiers
- **Comprehensive Feature Set**: 13 different student attributes
- **Real-time Predictions**: Instant grade class predictions based on input parameters
- **Data Visualization**: Exploratory data analysis with visualizations

## 📊 Dataset Features

The model uses the following features to predict student performance:

- **Age**: Student's age (10-25 years)
- **Gender**: Male/Female
- **Ethnicity**: Group A, B, C, D, E
- **Parental Education**: High School, Some College, Bachelor's, Master's, PhD
- **Study Time Weekly**: Hours spent studying per week
- **Absences**: Number of absences
- **Tutoring**: Whether student receives tutoring (Yes/No)
- **Parental Support**: Level of parental support (Yes/No)
- **Extracurricular**: Participation in extracurricular activities (Yes/No)
- **Sports**: Participation in sports (Yes/No)
- **Music**: Participation in music activities (Yes/No)
- **Volunteering**: Participation in volunteering (Yes/No)
- **GPA**: Current Grade Point Average (0.0-4.0)

## 🏗️ Project Structure

```
task no 1/
├── app.py                          # Streamlit web application
├── data_preprocessing.ipynb        # Data analysis and preprocessing notebook
├── decision_tree_model.pkl         # Trained Decision Tree model
├── svm_model.pkl                   # Trained Random Forest model
├── Student_performance_data _.csv  # Student performance dataset
├── Report.docx                     # Project documentation
└── README.md                       # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Dependencies

Install the required packages:

```bash
pip install streamlit
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

## 💻 Usage

### Running the Web Application

1. Navigate to the project directory:
```bash
cd "task no 1"
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`)

### Using the Application

1. **Select Model**: Choose between "Decision Tree" or "Random Forest" from the sidebar
2. **Input Features**: Fill in the student information:
   - Age (10-25)
   - Gender (Male/Female)
   - Ethnicity (Group A-E)
   - Parental Education Level
   - Study Time Weekly (hours)
   - Number of Absences
   - Tutoring Status
   - Parental Support
   - Extracurricular Activities
   - Sports Participation
   - Music Activities
   - Volunteering
   - Current GPA
3. **Predict**: Click the "Predict Grade Class" button to get the prediction

## 🔬 Model Information

### Decision Tree Classifier
- **Algorithm**: Decision Tree Classification
- **Features**: 13 student attributes
- **Output**: Grade Class prediction (0-4 scale)

### Random Forest Classifier
- **Algorithm**: Random Forest Classification
- **Features**: 13 student attributes
- **Output**: Grade Class prediction (0-4 scale)

## 📈 Data Analysis

The project includes comprehensive data analysis in `data_preprocessing.ipynb`:

- **Data Exploration**: Understanding dataset structure and statistics
- **Missing Value Analysis**: Checking for data quality issues
- **Feature Engineering**: Data preprocessing and normalization
- **Visualization**: Creating charts and plots for insights
- **Correlation Analysis**: Understanding feature relationships
- **Model Training**: Training and evaluating multiple algorithms

## 🎓 Grade Class Categories

The model predicts students into the following grade categories:
- **Class 0**: Excellent performance
- **Class 1**: Good performance
- **Class 2**: Average performance
- **Class 3**: Below average performance
- **Class 4**: Poor performance

## 📝 Technical Details

### Data Preprocessing
- Handled missing values
- Normalized numerical features
- Encoded categorical variables
- Removed duplicate entries
- Feature scaling and normalization

### Model Performance
- Cross-validation for model evaluation
- Multiple evaluation metrics (accuracy, precision, recall, F1-score)
- Confusion matrix analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes and internship task completion.

## 👨‍💻 Author

Created as part of DEN internship Task No. 1

## 📞 Support

For questions or issues, please refer to the project documentation or contact the development team.

---

**Note**: This project is designed for educational purposes and demonstrates machine learning concepts in student performance prediction. 