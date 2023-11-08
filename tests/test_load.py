from prompt_systematic_review.load import load_hf_dataset


def test_arxiv_source():
    # Small dataset for testing
    dataset = load_hf_dataset("rotten_tomatoes")
    assert len(dataset) == 3
    assert len(dataset["train"]) == 8530
    assert len(dataset["validation"]) == 1066
    assert len(dataset["test"]) == 1066
    assert dataset["train"].features["text"].dtype == "string"
    assert dataset["train"].features["label"].dtype == "int64"
    assert dataset["train"].features["label"].num_classes == 2
    assert dataset["train"][0]["label"] == 1
    assert (
        dataset["train"][0]["text"]
        == "the rock is destined to be the 21st century's new \" conan \" and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal ."
    )
