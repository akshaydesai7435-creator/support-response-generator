# Support Response Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/7evhnqgg8v/badge.svg)](https://ask.thili.ai/b/7evhnqgg8v)



A small command-line next-word predictor for retail customer-support responses. It trains a trigram (3-gram) language model on a small **synthetic, anonymized sample corpus** and uses add-one (Laplace) smoothing to estimate next-word probabilities.

## Requirements

- Python 3.10 or newer
- No API key, hosted model, or additional Python package is required for the n-gram implementation.

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Check the CLI:

```bash
python -m app.cli hello
```

Get next-word suggestions and a generated continuation:

```bash
python -m app.cli suggest "thank you for" --count 3 --words 8
```

Start an interactive session (type `quit` to exit):

```bash
python -m app.cli interactive
```

Run tests:

```bash
python -m pytest -q
```

## How it works

1. **Corpus preparation:** `data/support_corpus.txt` contains synthetic customer-support messages. The preprocessing pipeline masks common email addresses, phone numbers, and order/ticket identifiers before tokenization.
2. **N-gram model:** the trigram model counts two-word contexts and their following words. Laplace smoothing adds one to each candidate count, avoiding zero probabilities.
3. **Text generation:** the model ranks likely next words and can repeatedly append the most likely word to form a short continuation. It backs off to shorter context suffixes when a context has not appeared in training.
4. **CLI:** `suggest` handles a phrase supplied on the command line; `interactive` lets an agent enter multiple phrases.

The sample corpus is synthetic and contains no real customer records. For real deployment, replace it with an appropriately anonymized and authorized dataset.
