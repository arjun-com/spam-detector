import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

class SpamDetectionModel():
    def __init__(self, CSV_PATH):
        df = pd.read_csv(CSV_PATH, sep = "\t", header = None, names = ["label_str", "text"], quoting = 3)
        df["label"] = np.where(df["label_str"] == "spam", 1, 0)
        df = df.drop(columns=["label_str"])

        self.X = df["text"].astype(str)
        self.y = df["label"].astype(int)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 100, stratify = self.y)
        # stratify = y -> Ensures that the training and testing datasets have an equal distrubution of y values.

        self.train_df = pd.DataFrame({"text": self.X_train, "label": self.y_train})
        self.test_df = pd.DataFrame({"text": self.X_test, "label": self.y_test})

        self.pipe = Pipeline(
            steps=[
                (
                    "vectorization",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode", # Converts special letters ( like é to e ) into their english alphabet.
                        ngram_range=(1, 2), # Checks for unigram ( one word tokens like 'free' ) and bigram ( two word token 'shop now' )
                        # For min_df and max_df if the value is greater than 1 it assumes you're referring to number of docs, otherwise it assumes fraciton of docs.
                        min_df=2, # Each token should appear in atleast 2 docs.
                        max_df=0.95, # Each token should not appear in more than 95% of all docs.
                        sublinear_tf=True,
                    ),
                ),
                (
                    "logistic_regression",
                    LogisticRegression(
                        solver="liblinear",
                        C=3, # Inverse of regularization strength. Regularization strength is how much the model is penalized for overfitting.
                        max_iter=1000,
                        random_state=110,
                    ),
                ),
            ]
        )

        self.pipe.fit(self.X_train, self.y_train)

        self.y_pred = self.pipe.predict(self.X_test)

        acc_score = accuracy_score(self.y_test, self.y_pred)
        prec, recall, f1, _ = precision_recall_fscore_support(self.y_test, self.y_pred, average = "binary")

        print(f"Accuracy: {acc_score:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print("\nClassification report:\n")
        print(classification_report(self.y_test, self.y_pred, digits=4))

    def predict(self, text):
        return self.pipe.predict([text])[0]
