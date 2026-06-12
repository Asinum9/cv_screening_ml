"""Train the machine learning model for the CV Screening System.

This script proves that the project is machine learning based because it:
1. Loads a labelled dataset.
2. Uses cleaned CV text as input data.
3. Trains a supervised Logistic Regression model.
4. Evaluates the model using standard classification metrics.
5. Saves the trained model and vectorizer for the Streamlit app.

Run from the project folder:
    python train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# Project folders and file paths.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "it_cv_synthetic_dataset_180.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_FILE = MODEL_DIR / "resume_classifier.pkl"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"
REPORT_FILE = OUTPUT_DIR / "evaluation_report.txt"
PREDICTIONS_FILE = OUTPUT_DIR / "predictions_sample.csv"


# The assessment requires these exact columns.
REQUIRED_COLUMNS = ["cleaned_text", "target_role"]


def load_dataset() -> pd.DataFrame:
    """Load and validate the training dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place it inside the data folder before running this script."
        )

    dataset = pd.read_csv(DATA_FILE)

    # Check that the dataset has the exact columns needed for training.
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            f"{missing_columns}\n"
            "Required columns are: cleaned_text and target_role."
        )

    # Remove rows where the CV text or target label is empty.
    dataset = dataset[REQUIRED_COLUMNS].copy()
    dataset["cleaned_text"] = dataset["cleaned_text"].fillna("").astype(str).str.strip()
    dataset["target_role"] = dataset["target_role"].fillna("").astype(str).str.strip()
    dataset = dataset[
        (dataset["cleaned_text"] != "")
        & (dataset["target_role"] != "")
    ]

    if dataset.empty:
        raise ValueError("No usable rows found after removing empty text or label rows.")

    return dataset


def train_model() -> None:
    """Train, evaluate, and save the CV screening ML model."""
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset()

    # X is the input feature: cleaned CV text.
    # y is the supervised label: the correct target role.
    X = dataset["cleaned_text"]
    y = dataset["target_role"]

    print(f"Dataset loaded successfully: {len(dataset)} usable rows.")
    print(f"Target roles found: {sorted(y.unique())}")

    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Convert text into numerical machine learning features using TF-IDF.
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a supervised Logistic Regression classifier.
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train_tfidf, y_train)

    # Predict labels for the test set.
    y_pred = model.predict(X_test_tfidf)

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)

    # Save the trained model and vectorizer so app.py can use them later.
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    # Save a detailed evaluation report for the project evidence folder.
    report_text = (
        "Machine Learning CV Screening Evaluation Report\n"
        "================================================\n\n"
        "This report proves the project is ML-based because a supervised\n"
        "Logistic Regression model was trained using TF-IDF text features.\n\n"
        f"Dataset file: {DATA_FILE}\n"
        f"Input feature X: cleaned_text\n"
        f"Target label y: target_role\n"
        f"Total usable rows: {len(dataset)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n\n"
        "Model Configuration\n"
        "-------------------\n"
        "Vectorizer: TfidfVectorizer(max_features=5000, ngram_range=(1, 2))\n"
        "Classifier: LogisticRegression(max_iter=1000, class_weight='balanced')\n\n"
        "Evaluation Metrics\n"
        "------------------\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision (weighted): {precision:.4f}\n"
        f"Recall (weighted): {recall:.4f}\n"
        f"F1-score (weighted): {f1:.4f}\n\n"
        "Classification Report\n"
        "---------------------\n"
        f"{report}\n"
        "Confusion Matrix Labels\n"
        "-----------------------\n"
        f"{list(model.classes_)}\n\n"
        "Confusion Matrix\n"
        "----------------\n"
        f"{matrix}\n"
    )
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    # Save sample predictions so the results can be inspected in a spreadsheet.
    predictions_sample = pd.DataFrame({
        "cleaned_text_preview": X_test.reset_index(drop=True).str.slice(0, 180),
        "actual_target_role": y_test.reset_index(drop=True),
        "predicted_target_role": pd.Series(y_pred),
    })
    predictions_sample.to_csv(PREDICTIONS_FILE, index=False)

    print("\nTraining completed successfully.")
    print("This project is machine learning based: TF-IDF features + Logistic Regression model.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved vectorizer to: {VECTORIZER_FILE}")
    print(f"Saved evaluation report to: {REPORT_FILE}")
    print(f"Saved sample predictions to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    train_model()
