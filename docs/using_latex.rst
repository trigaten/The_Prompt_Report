=========================
Using LaTeX
=========================

LaTeX is a typesetting language commonly used for creating documents with complex formatting, such as research papers 
and scientific articles. For our collaborative document editing, we will be using Overleaf, an online LaTeX editor that 
allows multiple people to work on a document simultaneously. If you need access to the LaTeX repository for our project, 
please don't hesitate to reach out to me or your sub team lead.

Here are some basic formatting tips with LaTeX:

1. **Section Headings**: You can create section headings using LaTeX by starting a line with a series of `#` symbols. The number of `#` symbols determines the section level. For example, `#` is a top-level section, `##` is a subsection, and so on.

2. **Italics and Bold**: To make text italic, you can use `\\textit{}` and to make text bold, you can use `\\textbf{}`. For example, `\\textit{italic text}` and `\\textbf{bold text}`.


Here are some tips for our project:


1. Cite everything! Every time you refer to information from an external source, you need to cite it. Here is how to do so:

Say you want to cite Chain of Thought: https://arxiv.org/abs/2201.11903

- You would go its arXiv page and look for a pair of quotes kind of on bottom right of page. You might need to turn on Bibliographic Explorer. You can also just click Export BibTeX Citation.

- Copy the citation, which should look like this:

.. code:: text

    @misc{wei2023chainofthought,
        title={Chain-of-Thought Prompting Elicits Reasoning in Large Language Models}, 
        author={Jason Wei and Xuezhi Wang and Dale Schuurmans and Maarten Bosma and Brian Ichter and Fei Xia and Ed Chi and Quoc Le and Denny Zhou},
        year={2023},
        eprint={2201.11903},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }

- Paste it at the bottom of custom.bib in the Overleaf document. Make sure you don't mess up any other citations.

- Then, you can take the citation title (wei2023chainofthought) and use it in the paper itself. If I were on the agents team, I would go to the 80-agents.tex file then add \\cite{wei2023chainofthought} where I wanted the citation.

- If you need to start a sentence with a citation, use \\citet{wei2023chainofthought} instead.



Learn more here: `Link to LaTeX Documentation <https://www.overleaf.com/learn>`_


