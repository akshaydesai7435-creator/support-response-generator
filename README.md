# Support Response Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/7evhnqgg8v/badge.svg)](https://ask.thili.ai/b/7evhnqgg8v)

This Support Response Generator is a CLI tool designed for retail customer support agents. It leverages a next-word prediction model, trained on a corpus of previous support interactions, to offer real-time suggestions, streamlining the drafting of common queries and improving response efficiency.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that helps customer support agents quickly draft responses by generating next-word suggestions based on past support interactions.

**Done when.** The tool successfully processes a corpus of customer support interactions, implements an n-gram model with smoothing, generates coherent response continuations, and provides a functional command-line interface for agents to get suggestions.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare an anonymized corpus of customer support interactions.**  
      Hint: Use text preprocessing and tokenization techniques to clean and structure the data.
- [ ] **2. Implement the n-gram model with smoothing.**  
      Hint: Implement n-gram counting and apply a smoothing technique like Laplace smoothing.
- [ ] **3. Create a text generation function to suggest response continuations.**  
      Hint: Develop a function that uses the n-gram probabilities to select and suggest the most probable next words.
- [ ] **4. Build a command-line interface for agents to get suggestions.**  
      Hint: Utilize standard input/output operations to create an interactive command-line experience.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the user to specify the number of next-word suggestions to display.
- Implement a feature to save the generated response to a text file.
- Add a simple 'undo' or 'backspace' functionality to modify the current input phrase.
