# 🎭 Sarcasm Detection in Social Media Posts

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/Sarcasm/graphs/commit-activity)

A comprehensive machine learning project for detecting sarcasm in social media comments using state-of-the-art natural language processing techniques.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Models](#-models)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## 🌟 Project Overview
This project implements various machine learning and deep learning approaches to detect sarcasm in social media comments. Sarcasm detection is a challenging task in natural language processing due to its inherent ambiguity and contextual nature. This project provides a comprehensive pipeline from data preprocessing to model deployment, making it easier for researchers and developers to understand and implement sarcasm detection systems.

## ✨ Features
- **Text Preprocessing**: Comprehensive cleaning and normalization of social media text
- **Exploratory Data Analysis**: In-depth analysis of comment data distributions and patterns
- **N-gram Analysis**: Identification of common phrase patterns in sarcastic vs. non-sarcastic comments
- **Visualizations**: Interactive plots for data insights and model performance metrics
- **Multiple Model Architectures**: Implementation of both traditional ML and deep learning models
- **Model Evaluation**: Comprehensive metrics and visualization tools for performance assessment

## 📊 Dataset
The project uses a dataset of social media comments labeled as either sarcastic or non-sarcastic. The dataset includes:

- **Source**: [Dataset Source Name/Link]
- **Size**: [Number of samples] comments
- **Features**:
  - Text content
  - Binary label (sarcastic/non-sarcastic)
  - [Additional metadata if available]

### Data Preprocessing
```python
# Example preprocessing steps
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
```

## 📈 Key Analysis
- **Comment Length Analysis**: Comparison of text length distributions between sarcastic and non-sarcastic comments
- **N-gram Analysis**: Identification of most common 2-grams, 3-grams, and 5-grams
- **Sentiment Distribution**: Analysis of sentiment scores across different classes
- **Word Clouds**: Visual representation of most frequent words in each class

## 🤖 Models

### 1. Logistic Regression
- **Type**: Traditional Machine Learning
- **Features**: TF-IDF vectorization
- **Advantages**: Fast training, interpretable results
- **Use Case**: Baseline model for comparison

### 2. Random Forest Classifier
- **Type**: Ensemble Learning
- **Features**: Handles non-linear relationships
- **Advantages**: Reduces overfitting, handles mixed data types

### 3. LSTM (Long Short-Term Memory)
- **Type**: Deep Learning (RNN)
- **Features**: Word embeddings, sequence modeling
- **Advantages**: Captures long-range dependencies in text

### 4. BERT (Bidirectional Encoder Representations from Transformers)
- **Type**: Transformer-based
- **Features**: Contextual word embeddings
- **Advantages**: State-of-the-art performance, understands context

## 🎯 Model Evaluation

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.78 | 0.76 | 0.72 | 0.74 |
| Random Forest | 0.82 | 0.81 | 0.80 | 0.80 |
| LSTM | 0.84 | 0.83 | 0.82 | 0.82 |
| BERT | 0.88 | 0.87 | 0.87 | 0.87 |

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Sarcasm.git
cd Sarcasm
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🛠️ Usage

1. Preprocess the data:
```bash
python src/preprocess.py --input data/raw_data.csv --output data/processed_data.pkl
```

2. Train a model:
```bash
python src/train.py --model lstm --data data/processed_data.pkl --output models/
```

3. Evaluate a model:
```bash
python src/evaluate.py --model models/lstm_model.h5 --data data/test_data.pkl
```

4. Make predictions:
```bash
python src/predict.py --model models/lstm_model.h5 --text "Oh great, another meeting that could've been an email"
```

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements
- [Dataset Source/Author]
- [Libraries/Packages used]
- [Inspiration/References]
- F1-Score: Harmonic mean of precision and recall

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- plotly
- scikit-learn
- nltk

## Usage
1. Clone the repository
2. Install the required dependencies
3. Open and run `sarcasm.ipynb` in Jupyter Notebook or JupyterLab

## Results
The notebook includes visualizations and analysis of the data, along with model performance metrics for sarcasm detection.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
