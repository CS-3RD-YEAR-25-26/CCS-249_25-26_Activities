"""
Naïve Bayes using Scikit-Learn
==============================
Trains a Multinomial Naïve Bayes classifier on the same small dataset
using CountVectorizer and scikit-learn's MultinomialNB.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
DOCUMENTS = [
    "Free money now!!!",
    "Hi mom, how are you?",
    "Lowest price for your meds",
    "Are we still on for dinner?",
    "Win a free iPhone today",
    "Let's catch up tomorrow at the office",
    "Meeting at 3 PM tomorrow",
    "Get 50% off, limited time!",
    "Team meeting in the office",
    "Click here for prizes!",
    "Can you send the report?",
]

LABELS = [
    "SPAM", "HAM", "SPAM", "HAM", "SPAM", "HAM",
    "HAM", "SPAM", "HAM", "SPAM", "HAM",
]

TEST_SENTENCES = [
    "Limited offer, click here!",
    "Meeting at 2 PM with the manager.",
]


# ---------------------------------------------------------------------------
# 2. Vectorization
# ---------------------------------------------------------------------------
# token_pattern=r"(?u)\b[a-zA-Z]+\b" keeps only alphabetic tokens,
# matching the manual tokenizer closely.
vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-zA-Z]+\b", lowercase=True)
X_train = vectorizer.fit_transform(DOCUMENTS)

print("Vocabulary:")
print(sorted(vectorizer.vocabulary_.keys()))
print(f"\nDocument-term matrix shape: {X_train.shape}")


# ---------------------------------------------------------------------------
# 3. Train Multinomial Naïve Bayes
# ---------------------------------------------------------------------------
# alpha=1.0 is the default Laplace smoothing used in the manual version.
clf = MultinomialNB(alpha=1.0)
clf.fit(X_train, LABELS)

print("\nClass priors (log):")
for cls, log_prior in zip(clf.classes_, clf.class_log_prior_):
    print(f"  log P({cls}) = {log_prior:.6f}")


# ---------------------------------------------------------------------------
# 4. Predict test sentences
# ---------------------------------------------------------------------------
X_test = vectorizer.transform(TEST_SENTENCES)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print("\nScikit-Learn MultinomialNB Classification")
print("=" * 60)
for sentence, pred, probs in zip(TEST_SENTENCES, predictions, probabilities):
    print(f"\nSentence: {sentence}")
    print(f"Predicted class: {pred}")
    print("Class probabilities:")
    for cls, prob in zip(clf.classes_, probs):
        print(f"  P({cls}) = {prob:.6f}")
