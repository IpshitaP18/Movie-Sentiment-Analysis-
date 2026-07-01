import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

df = pd.read_csv("dataset/cleaned_IMDB_Dataset.csv")

sentiment_count=df['sentiment'].value_counts()
plt.bar(sentiment_count.index, sentiment_count.values)
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.title("Distribution of Positive and Negative Reviews")
plt.savefig("images/Sentiment_Distribution.png")
plt.show()

X = df["clean_review"]
y = df["sentiment"]

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Multinomial Naive Bayes":
        MultinomialNB(),

    "Support Vector Machine":
        LinearSVC(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

results = []
best_accuracy = 0
best_model = None
best_model_name = ""

for name, model in models.items():
    print("\nTraining:", name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="positive"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="positive"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="positive"
    )

    results.append(
        [
            name,
            accuracy,
            precision,
            recall,
            f1
        ]
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

results_df = pd.DataFrame(
    results,
    columns=[
        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nModel Comparison:")
print(results_df)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

best_prediction = best_model.predict(X_test)
cm = confusion_matrix(
    y_test,
    best_prediction
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - " + best_model_name)
plt.savefig("images/confusion_matrix.png")
plt.show()

with open("models/sentiment_model.pkl","wb") as file:
    pickle.dump(
        best_model,
        file
    )

with open("models/tfidf.pkl","wb") as file:
    pickle.dump(
        tfidf,
        file
    )
    
record_count = len(df)
with open("models/metadata.pkl", "wb") as file:
    pickle.dump(record_count, file)

print("\nBest Model:")
print(best_model_name)
print("\nModel Saved Successfully!")
