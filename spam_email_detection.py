import pandas as pd

df = pd.read_csv("spam.csv", encoding="latin-1")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nCategory Counts:")
print(df["Category"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

df["Category"] = df["Category"].map({
    "ham": 0,
    "spam": 1
})

print(df.head())

X = df["Message"]

y = df["Category"]

print("Messages:", X.shape)
print("Lables", y.shape)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)

print("New Shape:", X.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model trained successfully")

from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)