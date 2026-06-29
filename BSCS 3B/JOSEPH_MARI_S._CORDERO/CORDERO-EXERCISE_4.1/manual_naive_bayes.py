"""
Manual Naïve Bayes Implementation
=================================
Implements a simple Multinomial Naïve Bayes text classifier from scratch.
"""

import re
import math
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
DATASET = [
    ("Free money now!!!", "SPAM"),
    ("Hi mom, how are you?", "HAM"),
    ("Lowest price for your meds", "SPAM"),
    ("Are we still on for dinner?", "HAM"),
    ("Win a free iPhone today", "SPAM"),
    ("Let's catch up tomorrow at the office", "HAM"),
    ("Meeting at 3 PM tomorrow", "HAM"),
    ("Get 50% off, limited time!", "SPAM"),
    ("Team meeting in the office", "HAM"),
    ("Click here for prizes!", "SPAM"),
    ("Can you send the report?", "HAM"),
]

TEST_SENTENCES = [
    "Limited offer, click here!",
    "Meeting at 2 PM with the manager.",
]


def tokenize(text):
    """
    Convert a raw string into a list of lowercase alphabetic tokens.
    Punctuation and numbers are removed for the manual model.
    """
    # Keep only alphabetic characters, collapse multiple spaces, lowercase.
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.lower().split()


# ---------------------------------------------------------------------------
# 2. Bag of Words
# ---------------------------------------------------------------------------
def build_bag_of_words(dataset, label=None):
    """
    Build a Bag-of-Words counter.

    If `label` is given, count only documents belonging to that class.
    Otherwise count every document in the dataset.
    """
    bow = Counter()
    for doc, cls in dataset:
        if label is None or cls == label:
            bow.update(tokenize(doc))
    return bow


def print_bag_of_words(bow, title="Bag of Words"):
    print(f"\n{title}")
    print("-" * len(title))
    for word, count in sorted(bow.items()):
        print(f"  {word:<12} {count}")
    print(f"  {'TOTAL':<12} {sum(bow.values())}")


# ---------------------------------------------------------------------------
# 3. Prior probabilities
# ---------------------------------------------------------------------------
def compute_priors(dataset):
    """Return P(class) for each class label."""
    total = len(dataset)
    class_counts = Counter(cls for _, cls in dataset)
    return {cls: count / total for cls, count in class_counts.items()}


# ---------------------------------------------------------------------------
# 4. Likelihoods with Laplace smoothing
# ---------------------------------------------------------------------------
def compute_likelihoods(dataset, vocab):
    """
    Compute P(word | class) for every word in the vocabulary and each class.
    Uses add-one (Laplace) smoothing.
    """
    classes = sorted({cls for _, cls in dataset})
    likelihoods = {}

    for cls in classes:
        class_bow = build_bag_of_words(dataset, label=cls)
        total_words = sum(class_bow.values())
        denom = total_words + len(vocab)

        likelihoods[cls] = {}
        for word in vocab:
            count = class_bow.get(word, 0)
            likelihoods[cls][word] = (count + 1) / denom

    return likelihoods


def print_likelihoods(likelihoods, vocab):
    classes = sorted(likelihoods.keys())
    print("\nLikelihoods P(word | class)  (Laplace smoothing, alpha=1)")
    print("-" * 60)
    header = f"{'word':<15}" + "".join(f"{c:<15}" for c in classes)
    print(header)
    print("-" * len(header))
    for word in sorted(vocab):
        row = f"{word:<15}"
        for cls in classes:
            row += f"{likelihoods[cls][word]:<15.6f}"
        print(row)


# ---------------------------------------------------------------------------
# 5. Classifier
# ---------------------------------------------------------------------------
def predict_manual(text, priors, likelihoods, vocab):
    """
    Predict the class of `text` using log probabilities.
    Unknown words are ignored because Laplace smoothing already accounts
    for them through the vocabulary size.
    """
    tokens = [t for t in tokenize(text) if t in vocab]

    scores = {}
    for cls, prior in priors.items():
        log_prob = math.log(prior)
        for token in tokens:
            log_prob += math.log(likelihoods[cls][token])
        scores[cls] = log_prob

    predicted = max(scores, key=scores.get)
    return predicted, scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # a. Bag of Words
    full_bow = build_bag_of_words(DATASET)
    spam_bow = build_bag_of_words(DATASET, label="SPAM")
    ham_bow = build_bag_of_words(DATASET, label="HAM")

    print_bag_of_words(full_bow, "Bag of Words (full dataset)")
    print_bag_of_words(spam_bow, "Bag of Words (SPAM)")
    print_bag_of_words(ham_bow, "Bag of Words (HAM)")

    vocabulary = set(full_bow.keys())

    # b. Priors
    priors = compute_priors(DATASET)
    print("\nPrior Probabilities")
    print("-------------------")
    for cls, p in sorted(priors.items()):
        print(f"  P({cls}) = {p:.4f}")

    # c. Likelihoods
    likelihoods = compute_likelihoods(DATASET, vocabulary)
    print_likelihoods(likelihoods, vocabulary)

    # d. Classify test sentences
    print("\nManual Naïve Bayes Classification")
    print("=" * 60)
    for sentence in TEST_SENTENCES:
        pred, scores = predict_manual(sentence, priors, likelihoods, vocabulary)
        print(f"\nSentence: {sentence}")
        print(f"Tokens  : {tokenize(sentence)}")
        for cls in sorted(scores):
            print(f"  log P({cls}) = {scores[cls]:.6f}")
        print(f"  Predicted class: {pred}")
