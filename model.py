import os
import re
import string
import pickle
import warnings
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used")
os.makedirs('saved_model', exist_ok=True)


def load_dataset(path="IMDB Dataset.csv", sample_size=2000):
    df = pd.read_csv(path).sample(sample_size, random_state=42)
    print(f"Dataset loaded: {df.shape}")
    return df


def load_stopwords():
    url = 'https://raw.githubusercontent.com/Alir3z4/stop-words/master/english.txt'
    return set(pd.read_csv(url, header=None)[0])


stopwords = load_stopwords()
punctuations = set(string.punctuation)


def fast_tokenizer(text):
    tokens = re.findall(r'\b\w\w+\b', text.lower())
    return [word for word in tokens if word not in stopwords and word not in punctuations]


class TextCleaner(TransformerMixin):
    def transform(self, X, **transform_params):
        return [text.strip().lower() for text in X]

    def fit(self, X, y=None, **fit_params):
        return self


def train_model(name, classifier, X_train, X_test, y_train, y_test, model_path):
    model = Pipeline([
        ("cleaner", TextCleaner()),
        ("vectorizer", TfidfVectorizer(tokenizer=fast_tokenizer)),
        ("classifier", classifier)
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cr = classification_report(y_test, preds, output_dict=True)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.2%}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, preds))

    pickle.dump(model, open(model_path, 'wb'))

    return {
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Confusion Matrix": str(cm),
        "Precision_Pos": round(cr["positive"]["precision"], 2),
        "Recall_Pos": round(cr["positive"]["recall"], 2),
        "F1_Pos": round(cr["positive"]["f1-score"], 2),
        "Precision_Neg": round(cr["negative"]["precision"], 2),
        "Recall_Neg": round(cr["negative"]["recall"], 2),
        "F1_Neg": round(cr["negative"]["f1-score"], 2),
    }


df = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=77)

results = []
models_info = [
    ("Logistic Regression", LogisticRegression(), "saved_model/LogisticRegression_model.sav"),
    ("Random Forest", RandomForestClassifier(n_estimators=100), "saved_model/RandomForest_model.sav"),
    ("Linear SVC", LinearSVC(), "saved_model/LinearSVC_model.sav")
]

for name, clf, path in models_info:
    results.append(train_model(name, clf, X_train, X_test, y_train, y_test, path))

pd.DataFrame(results).to_excel("evaluation_results.xlsx", index=False)
print("\nEvaluation results exported to evaluation_results.xlsx")

loaded_models = {
    "Logistic Regression": pickle.load(open("saved_model/LogisticRegression_model.sav", "rb")),
    "Random Forest": pickle.load(open("saved_model/RandomForest_model.sav", "rb")),
    "Linear SVC": pickle.load(open("saved_model/LinearSVC_model.sav", "rb"))
}


def predict_sentiment():
    text = entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Required", "Please enter a review.")
        return

    predictions = {
        model_name: model.predict([text])[0]
        for model_name, model in loaded_models.items()
    }

    output = "\n".join([f"{name}: {pred}" for name, pred in predictions.items()])
    result_label.config(text=output)


root = tk.Tk()
root.title("IMDB Review Sentiment Classifier")
root.geometry("650x420")
root.config(bg="#f7f7f7")

tk.Label(root, text="Enter IMDB Review:", font=("Arial", 16), bg="#f7f7f7").pack(pady=10)
entry = tk.Text(root, height=7, width=70, font=("Arial", 12))
entry.pack(padx=10)

tk.Button(root, text="Predict Sentiment", command=predict_sentiment,
          font=("Arial", 14), bg="#4CAF50", fg="white").pack(pady=12)

result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f7f7f7", fg="black")
result_label.pack(pady=10)

root.mainloop()