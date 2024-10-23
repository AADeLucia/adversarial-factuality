""" """

import re
from dataclasses import dataclass
from overrides import overrides
from typing import Union, Text, List, Dict, Optional, Callable, Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser

from langchain_interface.example_selectors import (
    ConstantExampleSelector,
    ExampleSelector
)
from langchain_interface.steps.step import Step, FewShotStep
from langchain_interface.instances.instance import LLMResponse, Instance

from ..utils.prompts import RELEVANCY_PROMPT


@dataclass(frozen=True, eq=True)
class RelevancyResponse(LLMResponse):
    relevancy_score: float
    
    
class RelevancyOutputParser(BaseOutputParser[RelevancyResponse]):

    def parse(self, text: Text) -> Dict:
        search_result = re.search(r"\[(.*?)\]", text)
        if search_result is None:
            return 0.0
        return RelevancyResponse(
            messages=text,
            relevancy_score=1.0 if search_result.group(1).strip() == "Foo" else 0.0,
        )
    
    @property
    def _type(self) -> Text:
        return "relevancy-scoring"
    
    
@Step.register("relevancy-scoring")
class RelevancyScoringStep(FewShotStep):
    """ Judge if a claim is relevant to a sentence at hand. """
    
    def __init__(
        self,
        example_selector: ExampleSelector = None,
    ):
        if example_selector is None:
            example_selector = ConstantExampleSelector()
            examples = [
                {
                    "question": "Tell me a bio of Quoc Le.",
                    "response": "After completing his Ph.D., Quoc Le joined Google Brain, where he has been working on a variety of deep learning projects. Quoc is well-respected by many of his peers, such as Geoffrey Hinton, who is an adjunct professor at the University of Montreal and teaches courses on deep learning.",
                    "statement": "Geoffrey Hinton is at the University of Montreal.",
                    "solution": "The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is Geoffrey Hinton. The phrase \"Quoc is well-respected by many of his peers, such as Geoffrey Hinton\" from the RESPONSE shows that the relationship between Quoc Le and Geoffrey Hinton is that they are peers. For this reason, the subjects Quoc Le and Geoffrey Hinton are [Foo].",
                },
                {
                    "question": "Tell me a bio of Quoc Le.",
                    "response": "After completing his Ph.D., Quoc Le joined Google Brain, where he has been working on a variety of deep learning projects. Geoffrey Hinton is an adjunct professor at the University of Montreal, where he teaches courses on deep learning.",
                    "statement": "Geoffrey Hinton is at the University of Montreal.",
                    "solution": "The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is Geoffrey Hinton. While both subjects seem to be related to deep learning, the RESPONSE does not contain any phrases that explain what the relationship between Quoc Le and Geoffrey Hinton is. Thus, the subjects Quoc Le and Geoffrey Hinton are [Not Foo]."
                }
            ]
            
            for example in examples:
                example_selector.add_example(example)
                
        super().__init__(example_selector=example_selector)
    
    @overrides
    def get_prompt_template(self) -> Runnable:

        input_example_prompt = "QUESTION:\n{question}\n\nRESPONSE:\n{response}\n\nSTATEMENT:\n{statement}",
        output_example_prompt = "SOLUTION:\n{solution}",
        
        example_prompt = ChatPromptTemplate.from_messages(
            ("human", input_example_prompt),
            ("ai", output_example_prompt),
        )
        
        fewshot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector,
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("human", RELEVANCY_PROMPT),
                ("ai", "Sure, please provide me with a statement and a question you want me to check relevancy for."),
                fewshot_prompt_template,
                ("human", input_example_prompt)
            ]
        )
        
        return prompt_template
    
    @overrides
    def get_output_parser(self) -> Runnable:
        return RelevancyOutputParser()