"""
Train Skip-gram with Negative Sampling on a Wikipedia article,
then evaluate the embedding model with intrinsic tests and custom test sets.

Requirements:
    pip install requests beautifulsoup4 nltk gensim scikit-learn scipy

Optional:
    python -m nltk.downloader punkt stopwords
"""

import re
import math
import json
import random
from collections import Counter
from typing import List, Tuple, Dict

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# Paste the Wikipedia link you want to use here. 
# The article should be reasonably long (at least a few thousand words) for good results.
WIKI_URL = "https://en.wikipedia.org/wiki/Badminton"
RANDOM_SEED = 42


def ensure_nltk():
    resources = ["punkt", "punkt_tab"]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r)


def fetch_wikipedia_article(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SGNS-Training/1.0)"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    # Extract main content text from the Wikipedia page
    soup = BeautifulSoup(resp.text, "html.parser")

    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div is None:
        raise ValueError("Could not find Wikipedia article content.")

    paragraphs = content_div.find_all(["p", "li"])
    text_blocks = []

    for p in paragraphs:
        txt = p.get_text(" ", strip=True)
        if txt:
            text_blocks.append(txt)

    text = "\n".join(text_blocks)

    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> List[List[str]]:
    sentences = sent_tokenize(text)

    processed = []
    for sent in sentences:
        sent = sent.lower()
        sent = re.sub(r"[^a-z0-9\-\s]", " ", sent)
        sent = re.sub(r"\s+", " ", sent).strip()
        if not sent:
            continue

        tokens = word_tokenize(sent)

        cleaned = []
        for tok in tokens:
            tok = tok.strip("-")
            if not tok:
                continue
            if tok.isdigit():
                continue
            if len(tok) < 2:
                continue
            cleaned.append(tok)

        if len(cleaned) >= 3:
            processed.append(cleaned)

    return processed


def corpus_stats(sentences: List[List[str]]) -> Dict[str, int]:
    flat = [w for s in sentences for w in s]
    vocab = set(flat)
    return {
        "num_sentences": len(sentences),
        "num_tokens": len(flat),
        "vocab_size": len(vocab),
    }


def train_sgns(sentences: List[List[str]], window_size: int = 5) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=100, # What happens if we change this? Try 50, 200, 300 and see how it affects results.
        window=window_size,
        min_count=1,
        workers=4,
        sg=1,          # 0 = CBOW, 1 = skip-gram
        negative=10,   # negative sampling
        epochs=200,
        sample=1e-3,
        alpha=0.025,
        min_alpha=0.0007,
        seed=RANDOM_SEED,
    )
    return model


def has_word(model: Word2Vec, word: str) -> bool:
    return word in model.wv.key_to_index


def cosine(model: Word2Vec, w1: str, w2: str) -> float:
    v1 = model.wv[w1].reshape(1, -1)
    v2 = model.wv[w2].reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def evaluate_relatedness(model: Word2Vec, test_pairs: List[Tuple[str, str, float]]):
    gold = []
    pred = []
    covered = []

    for w1, w2, score in test_pairs:
        if has_word(model, w1) and has_word(model, w2):
            sim = cosine(model, w1, w2)
            gold.append(score)
            pred.append(sim)
            covered.append((w1, w2, score, sim))

    return {
        "covered_items": covered,
        "coverage": len(covered),
        "total": len(test_pairs),
    }


def evaluate_analogies(model: Word2Vec, analogies: List[Tuple[str, str, str, str]]):
    """
    Analogy format: a:b :: c:d
    Checks whether most_similar(positive=[b,c], negative=[a]) returns d.
    """
    covered = 0
    correct = 0
    details = []

    for a, b, c, d in analogies:
        if all(has_word(model, w) for w in [a, b, c, d]):
            covered += 1
            try:
                preds = model.wv.most_similar(positive=[b, c], negative=[a], topn=5)
                predicted_words = [w for w, _ in preds]
                hit = d in predicted_words
                correct += int(hit)
                details.append({
                    "analogy": f"{a}:{b}::{c}:?",
                    "expected": d,
                    "predictions": predicted_words,
                    "correct_in_top5": hit
                })
            except KeyError:
                pass

    accuracy = correct / covered if covered else float("nan")
    return {
        "coverage": covered,
        "total": len(analogies),
        "accuracy_top5": accuracy,
        "details": details
    }


def print_top_neighbors(model: Word2Vec, words: List[str], topn: int = 8):
    print("\n=== Nearest Neighbors ===")
    for word in words:
        if has_word(model, word):
            neighbors = model.wv.most_similar(word, topn=topn)
            print(f"\n{word}:")
            for neigh, score in neighbors:
                print(f"  {neigh:20s} {score:.4f}")
        else:
            print(f"\n{word}: [OOV]")


def plot_pca_vectors(model: Word2Vec, words: List[str], title: str = "PCA projection of Word2Vec embeddings"):
    """Visualize word vectors using PCA"""
    valid_words = [w for w in words if has_word(model, w)]
    if len(valid_words) < 2:
        print("Not enough in-vocab words for PCA.")
        return

    vectors = np.array([model.wv[w] for w in valid_words])
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(vectors)

    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6)

    for i, w in enumerate(valid_words):
        plt.annotate(w, (coords[i, 0], coords[i, 1]), fontsize=10, fontweight='bold')

    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved PCA plot to {title.replace(' ', '_').lower()}.png")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_nltk()

    print("Downloading Wikipedia article...")
    raw_text = fetch_wikipedia_article(WIKI_URL)

    print("Preprocessing text...")
    sentences = preprocess_text(raw_text)
    stats = corpus_stats(sentences)

    print("\n=== Corpus Stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # ============== MODEL 1: Window Size 5 ==============
    print("\n" + "="*70)
    print("MODEL 1: Training Skip-gram with Negative Sampling (Window=5)...")
    print("="*70)
    model_window5 = train_sgns(sentences, window_size=5)

    print("\nVocabulary size learned:", len(model_window5.wv))

    probe_words = [
        "badminton", "racket", "shuttlecock", "net", "court",
        "serve", "smash", "player", "match", "game"
    ]
    print_top_neighbors(model_window5, probe_words, topn=8)

    relatedness_test = [
        ("badminton", "racket", 0.95),
        ("badminton", "court", 0.90),
        ("serve", "smash", 0.85),
        ("player", "match", 0.80),
        ("shuttlecock", "net", 0.85),
        ("game", "match", 0.90),
        ("racket", "player", 0.75),
        ("court", "net", 0.80),
        ("badminton", "kitchen", 0.05),
        ("racket", "car", 0.02),
        ("shuttlecock", "ball", 0.50),
        ("serve", "player", 0.65),
    ]

    rel_results_w5 = evaluate_relatedness(model_window5, relatedness_test)

    print("\n=== Relatedness Test Set ===")
    print(f"Coverage: {rel_results_w5['coverage']}/{rel_results_w5['total']}")
    for w1, w2, gold, pred in rel_results_w5["covered_items"]:
        print(f"{w1:10s} - {w2:10s} | gold={gold:.2f} pred={pred:.4f}")

    analogy_test = [
        ("badminton", "racket", "tennis", "racket"),
        ("serve", "smash", "clear", "drop"),
        ("player", "match", "team", "game"),
        ("court", "net", "field", "ball"),
    ]

    analogy_results_w5 = evaluate_analogies(model_window5, analogy_test)

    print("\n=== Analogy Test Set ===")
    print(f"Coverage: {analogy_results_w5['coverage']}/{analogy_results_w5['total']}")
    print(f"Top-5 accuracy: {analogy_results_w5['accuracy_top5']}")
    for item in analogy_results_w5["details"]:
        print(json.dumps(item, ensure_ascii=False))

    print("\n=== Direct Similarity Checks ===")
    check_pairs = [
        ("badminton", "racket"),
        ("serve", "smash"),
        ("court", "net"),
        ("badminton", "kitchen"),
    ]
    for w1, w2 in check_pairs:
        if has_word(model_window5, w1) and has_word(model_window5, w2):
            print(f"{w1:10s} <-> {w2:10s}: {cosine(model_window5, w1, w2):.4f}")
        else:
            print(f"{w1:10s} <-> {w2:10s}: OOV")

    # ============== MODEL 2: Window Size 10 ==============
    print("\n" + "="*70)
    print("MODEL 2: Training Skip-gram with Negative Sampling (Window=10)...")
    print("="*70)
    model_window10 = train_sgns(sentences, window_size=10)

    print("\nVocabulary size learned:", len(model_window10.wv))
    print_top_neighbors(model_window10, probe_words, topn=8)

    rel_results_w10 = evaluate_relatedness(model_window10, relatedness_test)

    print("\n=== Relatedness Test Set (Window=10) ===")
    print(f"Coverage: {rel_results_w10['coverage']}/{rel_results_w10['total']}")
    for w1, w2, gold, pred in rel_results_w10["covered_items"]:
        print(f"{w1:10s} - {w2:10s} | gold={gold:.2f} pred={pred:.4f}")

    analogy_results_w10 = evaluate_analogies(model_window10, analogy_test)

    print("\n=== Analogy Test Set (Window=10) ===")
    print(f"Coverage: {analogy_results_w10['coverage']}/{analogy_results_w10['total']}")
    print(f"Top-5 accuracy: {analogy_results_w10['accuracy_top5']:.4f}" if not np.isnan(analogy_results_w10['accuracy_top5']) else f"Top-5 accuracy: nan")
    for item in analogy_results_w10["details"]:
        print(json.dumps(item, ensure_ascii=False))

    print("\n=== Direct Similarity Checks (Window=10) ===")
    for w1, w2 in check_pairs:
        if has_word(model_window10, w1) and has_word(model_window10, w2):
            print(f"{w1:10s} <-> {w2:10s}: {cosine(model_window10, w1, w2):.4f}")
        else:
            print(f"{w1:10s} <-> {w2:10s}: OOV")

    # ============== COMPARISON ==============
    print("\n" + "="*70)
    print("COMPARISON: Window=5 vs Window=10")
    print("="*70)
    print("\n--- Relatedness Test Performance ---")
    print(f"Window=5  Coverage: {rel_results_w5['coverage']}/{rel_results_w5['total']}")
    print(f"Window=10 Coverage: {rel_results_w10['coverage']}/{rel_results_w10['total']}")
    
    avg_sim_w5 = np.mean([pred for _, _, _, pred in rel_results_w5["covered_items"]]) if rel_results_w5["covered_items"] else 0
    avg_sim_w10 = np.mean([pred for _, _, _, pred in rel_results_w10["covered_items"]]) if rel_results_w10["covered_items"] else 0
    print(f"Window=5  Average Similarity: {avg_sim_w5:.4f}")
    print(f"Window=10 Average Similarity: {avg_sim_w10:.4f}")

    print("\n--- Analogy Test Performance ---")
    print(f"Window=5  Top-5 Accuracy: {analogy_results_w5['accuracy_top5']:.4f}" if not np.isnan(analogy_results_w5['accuracy_top5']) else f"Window=5  Top-5 Accuracy: nan")
    print(f"Window=10 Top-5 Accuracy: {analogy_results_w10['accuracy_top5']:.4f}" if not np.isnan(analogy_results_w10['accuracy_top5']) else f"Window=10 Top-5 Accuracy: nan")

    print("\nOLD:")
    print(
        "With window size 5, the model captures local context. "
        f"It achieved relatedness coverage of {rel_results_w5['coverage']}/{rel_results_w5['total']}, "
        f"average predicted similarity of {avg_sim_w5:.4f}, and analogy top-5 accuracy of "
        f"{analogy_results_w5['accuracy_top5'] if not np.isnan(analogy_results_w5['accuracy_top5']) else 'nan'}."
    )

    print("\nNEW:")
    print(
        "With window size 10, the model uses broader context from each sentence. "
        f"It achieved relatedness coverage of {rel_results_w10['coverage']}/{rel_results_w10['total']}, "
        f"average predicted similarity of {avg_sim_w10:.4f}, and analogy top-5 accuracy of "
        f"{analogy_results_w10['accuracy_top5'] if not np.isnan(analogy_results_w10['accuracy_top5']) else 'nan'}. "
        "From this change, a larger window usually improves topic-level semantic relatedness, "
        "while analogy gains can remain limited on a small single-article corpus."
    )

    # ============== PCA VISUALIZATION ==============
    print("\n" + "="*70)
    print("PCA VISUALIZATION")
    print("="*70)
    
    pca_words = [
        "badminton", "racket", "shuttlecock", "net", "court",
        "serve", "smash", "drop", "clear", "drive",
        "singles", "doubles", "player", "match", "game",
        "tournament", "champion", "points", "skill", "sport"
    ]
    
    print("\nGenerating PCA visualization for Window=5 model...")
    plot_pca_vectors(model_window5, pca_words, title="PCA Projection - Window Size 5")
    
    print("\nGenerating PCA visualization for Window=10 model...")
    plot_pca_vectors(model_window10, pca_words, title="PCA Projection - Window Size 10")

    # Save models
    model_window5.save("exercise_5_skipgram_window5.model")
    model_window10.save("exercise_5_skipgram_window10.model")
    print("\nSaved models to:")
    print("  - exercise_5_skipgram_window5.model")
    print("  - exercise_5_skipgram_window10.model")

    print("\nDone.")

main()