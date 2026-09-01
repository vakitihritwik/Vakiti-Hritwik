# 🏥 Patient Feedback Sentiment Analysis

An NLP portfolio project that classifies patient feedback as **positive, negative, or neutral** and demonstrates a practical text-analysis workflow.

## 🎯 Objective

Use natural language processing and machine learning to turn short patient comments into useful sentiment insights. This educational project does not use real patient records or provide medical advice.

## 🧰 Technologies

- Python
- Pandas
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression

## 📁 Project Structure

```text
patient-feedback-sentiment-analysis/
├── README.md
├── sentiment_analysis.py
└── requirements.txt
```

## 🚀 How to Run

```bash
pip install -r requirements.txt
python sentiment_analysis.py
```

The script uses a small, fictional dataset, performs text preprocessing, trains a TF-IDF + Logistic Regression classifier, prints evaluation metrics, and predicts sentiment for new sample comments.

## 🔬 NLP Workflow

1. Create a fictional feedback dataset.
2. Normalize text and remove common stop words.
3. Tokenize and lemmatize text using NLTK.
4. Convert text into TF-IDF features.
5. Train a Logistic Regression classifier.
6. Evaluate the model.
7. Predict sentiment for new feedback.

## 📌 Future Improvements

- Expand the dataset with properly licensed public feedback data.
- Compare Naive Bayes, SVM, and transformer-based models.
- Add confusion-matrix and sentiment-distribution visualizations.
- Build a Streamlit dashboard.
- Add multilingual sentiment analysis.

## 👨‍💻 Author

**Hritwik** — B.Tech CSE (Data Science) student

[GitHub](https://github.com/vakitihritwik)
