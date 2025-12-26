import numpy as np 
import pytest
from project import clean_text, generate_embedding, compute_similarity


def test_clean_text_basic():
    text = "This is a SIMPLE test sentence!"
    result = clean_text(text)

    assert isinstance(result, str)
    assert "test" in result
    assert "this" not in result


def test_clean_text_special_cases():
    text = "<p>Visit http://example.com in 2024</p>"
    result = clean_text(text)

    assert "http" not in result
    assert "2024" not in result
    assert "<p>" not in result


def test_clean_text_empty():
    assert clean_text("") == ""


def test_generate_embedding_basic():
    texts = ["machine learning", "data science"]
    vectors = generate_embedding(texts)

    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0


def test_generate_embedding_returns_numpy_array():
    vectors = generate_embedding(["resume", "job"])

    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0


def test_similarity_identical_text():
    texts = ["python developer", "python developer"]
    vectors = generate_embedding(texts)

    score = compute_similarity(vectors[0], vectors[1])
    assert score == pytest.approx(1.0, rel=1e-2)


def test_similarity_different_text():
    texts = ["python developer", "marketing manager"]
    vectors = generate_embedding(texts)

    score = compute_similarity(vectors[0], vectors[1])
    assert 0.0 <= score < 1.0


def test_similarity_type():
    texts = ["data science", "data analysis"]
    vectors = generate_embedding(texts)

    score = compute_similarity(vectors[0], vectors[1])
    assert isinstance(score, np.float32)
