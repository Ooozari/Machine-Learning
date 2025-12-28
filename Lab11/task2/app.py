from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for servers
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)  # ensure static dir exists


# Load built-in datasets
def load_dataset(name):
    if name == "iris":
        data = load_iris()
        return data.data, data.target
    elif name == "digits":
        data = load_digits()
        return data.data, data.target
    elif name == "wine":
        data = load_wine()
        return data.data, data.target
    else:
        raise ValueError("Unknown dataset name")


# Load model
def load_model(name):
    if name == "logistic":
        return LogisticRegression(max_iter=500)
    elif name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    elif name == "svm":
        return SVC()
    else:
        raise ValueError("Unknown model name")


@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    error = None

    if request.method == "POST":
        model_name = request.form.get("model")
        dataset_type = request.form.get("dataset_type")

        if not model_name or not dataset_type:
            error = "Model and dataset type are required."
            return render_template("index.html", results=None, error=error)

        # Load selected model
        try:
            model = load_model(model_name)
        except ValueError as e:
            error = str(e)
            return render_template("index.html", results=None, error=error)

        # Case 1: Built-in dataset
        if dataset_type == "builtin":
            dataset_name = request.form.get("dataset")
            if not dataset_name:
                error = "Please select a built-in dataset."
                return render_template("index.html", results=None, error=error)

            try:
                X, y = load_dataset(dataset_name)
            except ValueError as e:
                error = str(e)
                return render_template("index.html", results=None, error=error)

            dataset_used = dataset_name.upper()

        # Case 2: Uploaded CSV
        elif dataset_type == "csv":
            file = request.files.get("csv_file")
            if not file or file.filename == "":
                error = "Please upload a CSV file."
                return render_template("index.html", results=None, error=error)

            try:
                df = pd.read_csv(file)
            except Exception as e:
                error = f"Error reading CSV: {e}"
                return render_template("index.html", results=None, error=error)

            if df.shape[1] < 2:
                error = "CSV must have at least 2 columns (features + target)."
                return render_template("index.html", results=None, error=error)

            # Assume last column is target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            dataset_used = "Uploaded CSV"
        else:
            error = "Invalid dataset type."
            return render_template("index.html", results=None, error=error)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Feature Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train & Predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        rec = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        # Plot Metrics
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        values = [acc, prec, rec, f1]

        plt.figure()
        plt.bar(metrics, values)
        plt.ylim(0, 1)
        plt.title(f"Performance on {dataset_used}")
        plt.tight_layout()
        plt.savefig("static/metrics.png")
        plt.close()

        results = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "dataset": dataset_used,
            "plot_path": "static/metrics.png",
        }

    return render_template("index.html", results=results, error=error)


if __name__ == "__main__":
    app.run(debug=True)