import pytest
import numpy as np

from project import (
    clean_text,
    generate_embedding,
    compute_similarity
)

# -------------------------
# Tests for clean_text
# -------------------------

def test_clean_text_basic():
    text = "This is a simple TEST sentence!"
    result = clean_text(text)

    assert isinstance(result, str)
    assert "test" in result
    assert "this" not in result  # stopword removed


def test_clean_text_removes_numbers_and_urls():
    text = "Visit http://example.com in 2024"
    result = clean_text(text)

    assert "http" not in result
    assert "2024" not in result


def test_clean_text_empty_input():
    result = clean_text("")
    assert result == ""


def test_clean_text_html_tags():
    text = "<p>Hello World</p>"
    result = clean_text(text)

    assert "hello" in result
    assert "<p>" not in result


# -------------------------
# Tests for generate_embedding
# -------------------------

def test_generate_embedding_shape():
    texts = [
        "machine learning engineer",
        "data science engineer"
    ]

    vectors = generate_embedding(texts)

    assert vectors.shape[0] == 2  # 2 documents
    assert vectors.shape[1] > 0   # features exist


def test_generate_embedding_type():
    texts = ["resume text", "job description"]
    vectors = generate_embedding(texts)

    assert hasattr(vectors, "toarray")  # sparse matrix


# -------------------------
# Tests for compute_similarity
# -------------------------

def test_compute_similarity_identical_texts():
    texts = ["python developer", "python developer"]
    vectors = generate_embedding(texts)

    similarity = compute_similarity(vectors[0], vectors[1])

    assert similarity == pytest.approx(1.0, rel=1e-2)


def test_compute_similarity_different_texts():
    texts = ["python developer", "marketing manager"]
    vectors = generate_embedding(texts)

    similarity = compute_similarity(vectors[0], vectors[1])

    assert 0.0 <= similarity <= 1.0
    assert similarity < 1.0


def test_compute_similarity_output_type():
    texts = ["data science", "data analysis"]
    vectors = generate_embedding(texts)

    similarity = compute_similarity(vectors[0], vectors[1])

    assert isinstance(similarity, float)
