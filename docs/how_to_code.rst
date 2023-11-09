=========================
How to Code
=========================

Here are some tips on codebase maintenances/cleanliness:

- Always blacken your code. black is a Python codebase formatter, just run `black src` and/or `black tests` from the root before opening a PR. Otherwise, the precommit part of the CI pipeline will fail.

- Always add type hints to your functions/classes. In the below code example, the type hints are `int`, `List[str]`, and `List[Paper]`. You will have to import some of these types.

- Use Sphinx style docstrings, like the following. This allows Sphinx to generate the API reference part of this website.

.. code:: python

    class ArXivSource(PaperSource):
        """A class to represent a source of papers from ArXiv."""

        baseURL = "http://export.arxiv.org/api/query?search_query=all:"

        def getPapers(self, count: int, keyWords: List[str]) -> List[Paper]:
            """
            Get a list of papers from ArXiv that match the given keywords.

            :param count: The number of papers to retrieve.
            :type count: int
            :param keyWords: A list of keywords to match.
            :type keyWords: List[str]
            :return: A list of matching papers.
            :rtype: List[Paper]
            """

- Always write tests in the `tests/` directory. Every function you write should have a corresponding test that proves it works. You should even write tests before writing the function itself! This is called Test Driven Development (TDD).

- For API tests, add `@pytest.mark.API_test` like below. This allows them to be skipped during CI pipeline runs.

.. code:: python

    @pytest.mark.API_test
    def test_arxiv_source():