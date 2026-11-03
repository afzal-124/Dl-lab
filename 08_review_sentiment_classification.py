# Experiment 8: Classify text reviews as positive or negative
# Install: pip install scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

reviews = [
    "I loved this movie, it was fantastic",
    "The product is excellent and very useful",
    "Amazing experience, I really enjoyed it",
    "The service was wonderful",
    "Very happy with my purchase",
    "This product is terrible",
    "I hated this movie",
    "Very disappointing experience",
    "The product broke after one day",
    "Poor quality and bad service",
    "Absolutely brilliant and enjoyable",
    "I am satisfied with the quality",
    "The worst experience I have ever had",
    "Completely useless and disappointing",
    "Excellent service and great support",
    "The item is damaged and awful",
    "I would highly recommend this product",
    "I regret buying this product",
    "Fantastic quality and fast delivery",
    "The movie was boring and terrible",
]
labels = [1] * 10 + [0] * 10  # 1 = Positive, 0 = Negative

X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.25, random_state=42, stratify=labels
)

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000, random_state=42),
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=["Negative", "Positive"],
                            zero_division=0))

new_reviews = [
    "The movie was amazing and I loved it",
    "The product was awful and disappointing",
    "I am very happy with this purchase",
]
print("\nNew Review Predictions:")
for review, prediction in zip(new_reviews, model.predict(new_reviews)):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"\nReview: {review}\nSentiment: {sentiment}")
