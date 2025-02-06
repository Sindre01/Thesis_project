from langchain_core.example_selectors.base import BaseExampleSelector


class CoverageExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):

        pass
    