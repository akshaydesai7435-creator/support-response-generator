
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "support_corpus.txt"


DEFAULT_CORPUS = [
    "Hello, thank you for contacting customer support. I can help you with your order.",
    "Your order has been received and is being processed. You will receive a confirmation email shortly.",
    "Your order is on the way. You can track your delivery using the tracking link in your email.",
    "I am sorry for the delay. We are checking the shipment status and will update you soon.",
    "Thank you for your patience. Your refund has been initiated and may take five to seven business days.",
    "I understand your concern. Please share your order number so I can check the details.",
    "We apologize for the inconvenience. We have escalated this issue to our support team.",
    "Your payment was successful. The receipt has been sent to your registered email address.",
    "The item is currently out of stock. You can check the website for availability updates.",
    "You can return the item within the return period if it is unused and in its original packaging.",
    "Thank you for reaching out. Please let us know if you need any further assistance.",
    "I can help you update your delivery address before the order is shipped.",
    "Your replacement request has been approved. We will share the tracking details once it is dispatched.",
    "We are sorry that the product did not meet your expectations. We can help arrange a return.",
    "Your cancellation request has been received. The refund will be processed after confirmation.",
]

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
_ORDER_RE = re.compile(r"\b(?:order|ticket|case)[ #:-]*[A-Z0-9-]{4,}\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+", re.IGNORECASE)


def anonymize_text(text: str) -> str:
    """Mask common contact details and order/ticket identifiers."""
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _PHONE_RE.sub("<PHONE>", text)
    text = _ORDER_RE.sub("<ORDER_ID>", text)
    return text


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text, retaining words and numeric tokens."""
    return _TOKEN_RE.findall(anonymize_text(str(text)).lower())


def load_corpus(path: str | Path | None = None) -> list[str]:
    """Load one support interaction per non-empty line from a text file."""
    corpus_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    if not corpus_path.exists():
        return list(DEFAULT_CORPUS)
    return [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_corpus(corpus: Iterable[str] | None = None) -> list[list[str]]:
    """Anonymize, clean, and tokenize support interactions."""
    messages = load_corpus() if corpus is None else list(corpus)
    return [tokens for message in messages if (tokens := tokenize(message))]


class NGramModel:
    """An n-gram next-token model with add-one (Laplace) smoothing."""

    def __init__(self, n: int = 3):
        if n < 2:
            raise ValueError("n must be at least 2")
        self.n = n
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.vocabulary: set[str] = set()
        self._fitted = False

    def fit(self, sentences: Iterable[Sequence[str] | str]) -> "NGramModel":
        self.context_counts.clear()
        self.ngram_counts.clear()
        self.vocabulary.clear()
        for sentence in sentences:
            tokens = tokenize(sentence) if isinstance(sentence, str) else [str(t).lower() for t in sentence]
            if not tokens:
                continue
            self.vocabulary.update(tokens)
            padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for index in range(self.n - 1, len(padded)):
                ngram = tuple(padded[index - self.n + 1:index + 1])
                context, word = ngram[:-1], ngram[-1]
                self.context_counts[context] += 1
                self.ngram_counts[ngram] += 1
        self.vocabulary.add("</s>")
        self._fitted = True
        return self

    def predict(self, context: str | Sequence[str], top_k: int = 3) -> list[tuple[str, float]]:
        """Return (word, probability) pairs ordered from most to least likely."""
        if not self._fitted:
            raise RuntimeError("Fit the model before requesting predictions")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        tokens = tokenize(context) if isinstance(context, str) else [str(t).lower() for t in context]
        ctx = tuple(tokens[-(self.n - 1):]) if tokens else tuple()
        # Back off to a shorter suffix when the complete context was unseen.
        while ctx and self.context_counts[ctx] == 0:
            ctx = ctx[1:]
        candidates = sorted(self.vocabulary - {"<s>"})
        denominator = self.context_counts[ctx] + len(candidates)
        scored = []
        for word in candidates:
            count = self.ngram_counts[ctx + (word,)]
            probability = (count + 1) / denominator
            scored.append((word, probability))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]

    def generate(self, prompt: str, max_words: int = 12) -> str:
        """Append the most probable next words to a prompt."""
        if max_words < 0:
            raise ValueError("max_words cannot be negative")
        generated = tokenize(prompt)
        for _ in range(max_words):
            predictions = self.predict(generated, top_k=1)
            if not predictions:
                break
            next_word = predictions[0][0]
            if next_word == "</s>":
                break
            generated.append(next_word)
        return " ".join(generated)


def build_model(corpus: Iterable[Sequence[str] | str] | None = None, n: int = 3) -> NGramModel:
    """Prepare a corpus and train the n-gram model."""
    return NGramModel(n=n).fit(prepare_corpus() if corpus is None else corpus)


def suggest_next_words(prompt: str, model: NGramModel | None = None, top_k: int = 3) -> list[tuple[str, float]]:
    """Convenience function for getting next-word suggestions."""
    trained_model = model or build_model()
    return trained_model.predict(prompt, top_k=top_k)


# Milestone entry points, retained from the supplied starter scaffold.
def milestone_1():
    """Return the anonymized and tokenized sample support corpus."""
    return prepare_corpus()


def milestone_2():
    """Train and return a trigram model using Laplace smoothing at prediction time."""
    return build_model(n=3)


def milestone_3(prompt: str = "thank you for", top_k: int = 3):
    """Return likely next words and probabilities for a prompt."""
    return suggest_next_words(prompt, model=milestone_2(), top_k=top_k)


def milestone_4():
    """Start the interactive command-line interface."""
    from app.cli import main
    main(["interactive"])
