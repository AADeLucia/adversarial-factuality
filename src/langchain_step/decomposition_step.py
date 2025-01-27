"""Prompt-based scorer for the model.

Modeled after https://github.com/zipJiang/langchain-interface/blob/main/langchain_interface/steps/decomposition_step.py
but does not separate prompt and examples from the configuration.
"""

from dataclasses import dataclass
from overrides import overrides
from typing import Union, Text, List, Dict, Optional, Callable, Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain_core.output_parsers import BaseOutputParser

from langchain_interface.steps import (
    Step,
    FewShotStep
)
from langchain_interface.instances.instance import LLMResponse, Instance
from langchain_interface.example_selectors import ConstantExampleSelector, ExampleSelector


@dataclass(frozen=True, eq=True)
class DecompositionResponse(LLMResponse):
    claims: Text


class DecompositionOutputParser(BaseOutputParser[DecompositionResponse]):
    """Parse the output of the decomposition model.
    """

    def parse(self, text: Text) -> Dict:
        cleaned_text = text.strip()
        items = cleaned_text.split("\n")
        return DecompositionResponse(messages=text, claims=[item.replace('- ', "") for item in items])

    @property
    def _type(self) -> str:
        return "decompose"


# @Step.register("decompose")
class DecompositionStep(FewShotStep):
    """Break sentence into independent facts."""

    def __init__(
            self,
            system_prompt: str,
            input_example_prompt: str,
            examples: Optional[List] = None,
    ):
        self.system_prompt = system_prompt
        self.input_example_prompt = input_example_prompt
        if examples:
            example_selector = ConstantExampleSelector()
            for example in examples:
                example_selector.add_example(example)
        else:
            example_selector = None
        super().__init__(example_selector=example_selector)

    @overrides
    def get_prompt_template(self) -> Runnable:
        """
        input: input text to be decomposed.
        """
        few_shot_messages = []
        if self.system_prompt.strip() != "":
            few_shot_messages.append(("system", self.system_prompt))
        few_shot_messages.extend([
            ("human", self.input_example_prompt),
            ("ai", "{output}"),
        ])
        example_prompt = ChatPromptTemplate.from_messages(few_shot_messages)
        fewshot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector,
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                fewshot_prompt_template,
                ("human", self.input_example_prompt),
            ]
        )
        return prompt_template

    @overrides
    def get_output_parser(self) -> Runnable:
        return DecompositionOutputParser()
