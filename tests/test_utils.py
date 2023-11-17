# test_utils.py
from prompt_systematic_review.utils import process_paper_title


def test_process_paper_title():
    assert process_paper_title("Laws  of the\n Wildabeest") == "laws of the wildabeest"
    assert (
        process_paper_title("Laws --of the    \n Wildabeest")
        == "laws of the wildabeest"
    )
    assert process_paper_title("Broken-  tacos") == "broken tacos"
    assert process_paper_title("Right  Sword -  tacos") == "right sword tacos"
