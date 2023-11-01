# test_utils.py
from prompt_systematic_review.utils import process_paper_title



def test_process_paper_title():
    assert process_paper_title("Laws  of the\n Wildabeest") == "laws of the wildabeest"
