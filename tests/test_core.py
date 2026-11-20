"""Tests for the four Support Response Generator milestones."""
from app.core import NGramModel, milestone_1, milestone_2, milestone_3, tokenize


def test_milestone_1_prepares_anonymized_tokenized_corpus():
    corpus = milestone_1()
    assert corpus and all(isinstance(sentence, list) for sentence in corpus)
    assert all(sentence for sentence in corpus)
    assert tokenize("Contact me at person@example.com") == ["contact", "me", "at", "email"]


def test_milestone_2_trains_a_smoothed_ngram_model():
    model = milestone_2()
    predictions = model.predict("thank you for", top_k=3)
    assert len(predictions) == 3
    assert all(word and 0 < probability <= 1 for word, probability in predictions)
    assert abs(sum(probability for _, probability in model.predict("thank you for", top_k=len(model.vocabulary))) - 1.0) < 1e-9


def test_milestone_3_generates_next_word_suggestions():
    suggestions = milestone_3("thank you for", top_k=2)
    assert len(suggestions) == 2
    assert suggestions[0][1] >= suggestions[1][1]
    model = NGramModel(n=3).fit([["hello", "there"], ["hello", "customer"]])
    assert model.generate("hello", max_words=2).startswith("hello")


def test_milestone_4_cli_is_available(capsys):
    from app.cli import main
    main(["suggest", "thank you for", "--count", "2", "--words", "3"])
    output = capsys.readouterr().out
    assert "Next-word suggestions" in output
    assert "Suggested continuation" in output
