import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


DATA_PATH = "data/spam.csv"
MODEL_PATH = "spam_classifier.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"


def load_data(path: str) -> pd.DataFrame:
    """Load and clean the SMS spam dataset."""
    # The common Kaggle/UCI version uses latin-1 encoding and has extra
    # unnamed columns full of NaNs - drop those.
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label", "message"])
    return df


def build_and_evaluate(df: pd.DataFrame, model_name: str = "naive_bayes"):
    """Train a classifier and print evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # TF-IDF: turns raw text into weighted word-importance vectors.
    # stop_words='english' removes common filler words (the, is, at...)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "svm": LinearSVC(),
    }
    model = models[model_name]
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print(f"\n=== Results: {model_name} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    return model, vectorizer


def predict_message(model, vectorizer, message: str) -> str:
    """Classify a single new message."""
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    return "spam" if pred == 1 else "ham"


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} messages ({df['label'].sum()} spam, {len(df) - df['label'].sum()} ham)")

    # Try all three suggested algorithms and compare
    for name in ["naive_bayes", "logistic_regression", "svm"]:
        model, vectorizer = build_and_evaluate(df, model_name=name)

    # Save the last-trained model + vectorizer for reuse
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model to {MODEL_PATH} and vectorizer to {VECTORIZER_PATH}")

    # Quick manual test
    sample_messages = [
        "Congratulations! You've won a free ticket to Bahamas, call now!",
        "Hey, are we still on for lunch tomorrow?",
    ]
    for msg in sample_messages:
        print(f"'{msg}' -> {predict_message(model, vectorizer, msg)}")
