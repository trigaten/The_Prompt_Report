import collections
from prompt_systematic_review.automated_review import review_abstract_title_categorical
import pytest


@pytest.mark.API_test
def test_review_abstract_title_categorical():
    # 1st test
    review = review_abstract_title_categorical(
        "align and prompt: video-and-language pre-training with entity prompts",
        "Video-and-language pre-training has shown promising improvements on various downstream tasks. Most previous methods capture cross-modal interactions with a standard transformer-based multimodal encoder, not fully addressing the misalignment between unimodal video and text features. Besides, learning fine-grained visual-language alignment usually requires off-the-shelf object detectors to provide object information, which is bottlenecked by the detector’s limited vocabulary and expensive computation cost. In this paper, we propose Align and Prompt: a new video-and-language pre-training framework (ALPRO), which operates on sparsely-sampled video frames and achieves more effective cross-modal alignment without explicit object detectors. First, we introduce a video-text contrastive (VTC) loss to align unimodal video-text features at the instance level, which eases the modeling of cross-modal interactions. Then, we propose a novel visually-grounded pre-training task, prompting entity modeling (PEM), which learns fine-grained alignment between visual region and text entity via an entity prompter module in a self-supervised way. Finally, we pretrain the video-and-language transformer models on large webly-source video-text pairs using the proposed VTC and PEM losses as well as two standard losses of masked language modeling (MLM) and video-text matching (VTM). The resulting pre-trained model achieves state-of-the-art performance on both text-video retrieval and videoQA, outperforming prior work by a substantial margin. Implementation and pre-trained models are available at https://github.com/salesforce/ALPRO.",
        "gpt-4-1106-preview",
    )

    # Asserts that probability is in the dictionary of relevant reviews
    pDict = [
        "highly relevant",
        "somewhat relevant",
        "neutrally relevant",
        "somewhat irrelevant",
        "highly irrelevant",
    ]
    assert review.get("Probability") in pDict

    # Asserts that the abstract and the returned reasoning have common elements
    abstract = "Video-and-language pre-training has shown promising improvements on various downstream tasks. Most previous methods capture cross-modal interactions with a standard transformer-based multimodal encoder, not fully addressing the misalignment between unimodal video and text features. Besides, learning fine-grained visual-language alignment usually requires off-the-shelf object detectors to provide object information, which is bottlenecked by the detector’s limited vocabulary and expensive computation cost. In this paper, we propose Align and Prompt: a new video-and-language pre-training framework (ALPRO), which operates on sparsely-sampled video frames and achieves more effective cross-modal alignment without explicit object detectors. First, we introduce a video-text contrastive (VTC) loss to align unimodal video-text features at the instance level, which eases the modeling of cross-modal interactions. Then, we propose a novel visually-grounded pre-training task, prompting entity modeling (PEM), which learns fine-grained alignment between visual region and text entity via an entity prompter module in a self-supervised way. Finally, we pretrain the video-and-language transformer models on large webly-source video-text pairs using the proposed VTC and PEM losses as well as two standard losses of masked language modeling (MLM) and video-text matching (VTM). The resulting pre-trained model achieves state-of-the-art performance on both text-video retrieval and videoQA, outperforming prior work by a substantial margin. Implementation and pre-trained models are available at https://github.com/salesforce/ALPRO."
    result = collections.Counter(list(abstract.split(" "))) & collections.Counter(
        list(review.get("Reasoning").split(" "))
    )
    intersected_list = list(result.elements())

    assert len(intersected_list) > 0

    # 2nd test
    review = review_abstract_title_categorical(
        "hide and seek (has): a lightweight framework for prompt privacy protection",
        "Numerous companies have started offering services based on large language models (LLM), such as ChatGPT, which inevitably raises privacy concerns as users' prompts are exposed to the model provider. Previous research on secure reasoning using multi-party computation (MPC) has proven to be impractical for LLM applications due to its time-consuming and communication-intensive nature. While lightweight anonymization techniques can protect private information in prompts through substitution or masking, they fail to recover sensitive data replaced in the LLM-generated results. In this paper, we expand the application scenarios of anonymization techniques by training a small local model to de-anonymize the LLM's returned results with minimal computational overhead. We introduce the HaS framework, where 'H(ide)' and 'S(eek)' represent its two core processes: hiding private entities for anonymization and seeking private entities for de-anonymization, respectively. To quantitatively assess HaS's privacy protection performance, we propose both black-box and white-box adversarial models. Furthermore, we conduct experiments to evaluate HaS's usability in translation and classification tasks. The experimental findings demonstrate that the HaS framework achieves an optimal balance between privacy protection and utility.",
        "gpt-4-1106-preview",
    )
    # Asserts that probability is in the dictionary of relevant reviews
    pDict = [
        "highly relevant",
        "somewhat relevant",
        "neutrally relevant",
        "somewhat irrelevant",
        "highly irrelevant",
    ]
    assert review.get("Probability") in pDict

    """Asserts that the abstract and the returned reasoning have common elements to ensure that gpt model
    is using data from the abstract for the reasoning"""

    abstract = "Numerous companies have started offering services based on large language models (LLM), such as ChatGPT, which inevitably raises privacy concerns as users' prompts are exposed to the model provider. Previous research on secure reasoning using multi-party computation (MPC) has proven to be impractical for LLM applications due to its time-consuming and communication-intensive nature. While lightweight anonymization techniques can protect private information in prompts through substitution or masking, they fail to recover sensitive data replaced in the LLM-generated results. In this paper, we expand the application scenarios of anonymization techniques by training a small local model to de-anonymize the LLM's returned results with minimal computational overhead. We introduce the HaS framework, where 'H(ide)' and 'S(eek)' represent its two core processes: hiding private entities for anonymization and seeking private entities for de-anonymization, respectively. To quantitatively assess HaS's privacy protection performance, we propose both black-box and white-box adversarial models. Furthermore, we conduct experiments to evaluate HaS's usability in translation and classification tasks. The experimental findings demonstrate that the HaS framework achieves an optimal balance between privacy protection and utility."
    result = collections.Counter(list(abstract.split(" "))) & collections.Counter(
        list(review.get("Reasoning").split(" "))
    )
    intersected_list = list(result.elements())

    assert len(intersected_list) > 0
