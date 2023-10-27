# test_pipeline.py
import pytest
from prompt_systematic_review.pipeline import is_authenticated

def test_is_authenticated():
    assert is_authenticated() == False

def test_dummy():
    assert 1 == 1