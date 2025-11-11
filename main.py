import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

CSV_PATH = "./dataset/SMSSpamCollection"

def create_df():
    df = pd.read_csv(CSV_PATH, sep = "\t", header = None, names = ["label_str", "text"], quoting = 3)
    df["label"] = np.where(df["label_str"] == "spam", 1, 0)
    df.drop(columns=["label_str"])

    return df

df = create_df()

X = df["text"].astype(str)
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100, stratify = y)

train_df = pd.DataFrame({"text": X_train, "label": y_train})
test_df = pd.DataFrame({"text": X_test, "label": y_test})

pipe = Pipeline(
    steps=[
        (
            "vectorize",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 1),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "logistic regression",
            LogisticRegression(
                solver="liblinear",
                C=1.0,
                max_iter=1000,
                random_state=110,
            ),
        ),
    ]
)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average = "binary")

print(f"Accuracy: {acc_score:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, digits=4))
