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
)
from langchain_core.output_parsers import BaseOutputParser

from langchain_interface.steps import Step
from langchain_interface.instances.instance import LLMResponse, Instance


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
class DecompositionStep(Step):
    """Break sentence into independent facts."""

    def __init__(
        self,
        system_prompt: str,
        input_example_prompt: str,
    ):
    	self.system_prompt = system_prompt
    	self.input_example_prompt = input_example_prompt
    	super().__init__()

    @overrides
    def get_prompt_template(self) -> Runnable:
        """
        input: input text to be decomposed.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
            	("system", self.system_prompt),
                ("human", self.input_example_prompt)
            ]
        )

        return prompt_template

    @overrides
    def get_output_parser(self) -> Runnable:
        return DecompositionOutputParser()
