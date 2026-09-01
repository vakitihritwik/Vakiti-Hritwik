"""Patient feedback sentiment analysis using fictional educational data."""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


def download_nltk_resources():
    """Download the small NLTK resources required by this demo."""
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def build_dataset():
    """Return a fictional patient-feedback dataset."""
    rows = [
        ("The staff were kind and helpful", "positive"),
        ("The doctor explained everything clearly", "positive"),
        ("The appointment was smooth and comfortable", "positive"),
        ("The nurses were friendly and caring", "positive"),
        ("The clinic was clean and well organized", "positive"),
        ("I had a very good experience", "positive"),
        ("The waiting time was too long", "negative"),
        ("The staff were rude and unhelpful", "negative"),
        ("I was unhappy with the service", "negative"),
        ("The appointment felt rushed", "negative"),
        ("The clinic was difficult to navigate", "negative"),
        ("Communication from the staff was poor", "negative"),
        ("The visit was okay", "neutral"),
        ("The appointment was average", "neutral"),
        ("The waiting room was acceptable", "neutral"),
        ("The doctor answered my questions", "neutral"),
        ("The service met my expectations", "neutral"),
        ("The experience was neither good nor bad", "neutral"),
    ]
    return pd.DataFrame(rows, columns=["feedback", "sentiment"])


def preprocess(text, stop_words, lemmatizer):
    """Lowercase, remove punctuation/stop words, and lemmatize."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return " ".join(
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    )


def main():
    download_nltk_resources()
    data = build_dataset()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    data["clean_text"] = data["feedback"].apply(
        lambda text: preprocess(text, stop_words, lemmatizer)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_text"],
        data["sentiment"],
        test_size=0.33,
        random_state=RANDOM_STATE,
        stratify=data["sentiment"],
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions, zero_division=0))

    new_feedback = [
        "The nurses were very supportive",
        "The waiting time was frustrating",
        "The appointment was average",
    ]
    new_clean = [preprocess(text, stop_words, lemmatizer) for text in new_feedback]
    print("\nSample Predictions:")
    for text, sentiment in zip(new_feedback, model.predict(new_clean)):
        print(f"- {text} -> {sentiment}")


if __name__ == "__main__":
    main()
