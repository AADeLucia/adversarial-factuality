"""Adopting an LLM to do Evidential support scoring.
"""

from dataclasses import dataclass, field
import string
from typing import Text, Dict, List, Union, Optional, AsyncGenerator, Tuple
import os
from overrides import overrides
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from ..langchain_step.factscore_evidential_support_step import (
    FActScoreEvidentialSupportStep,
    FActScoreEvidentialSupportResponse,
)
from ..utils.instances import ScorerInstance
from .scorer import Scorer
from ..retriever.retriever import Retriever


# @dataclass(frozen=True, eq=True)
# class FActScoreQueryInstance(LLMQueryInstance):
#     topic: Text = field(default=None)
#     passages: List[Dict[Text, Text]] = field(default_factory=list)


@Scorer.register("llm-support")
class LLMSupportScorer(Scorer):

    __NAME__ = "llm-support"

    def __init__(
        self,
        model_name: Text,
        retriever: Retriever,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
        # retriever_batch_size: int = 256
    ):
        """ """
        super().__init__()

        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        # self._retriever_batch_size = retriever_batch_size

        # def _parse_input(instance: FActScoreQueryInstance) -> Dict[Text, Text]:
        #     """Generate the input dictionary for the LLM.
        #     """
        #     return {
        #         "parsed_passages": '\n\n'.join([f"Title: {passage['title']} Text: {passage['text']}" for passage in instance.passages]) + '\n\n',
        #         "topic": instance.topic,
        #         "input": instance.input
        #     }

        # def _parse_output(output: Text) -> float:
        #     """Parse the output of the LLM.
        #     """
        #     generated_answer = output.strip().lower()
        #     is_supported = 0.0

        #     if "true" in generated_answer or "false" in generated_answer:
        #         if "true" in generated_answer and "false" not in generated_answer:
        #             is_supported = 1.0
        #         elif "false" in generated_answer and "true" not in generated_answer:
        #             is_supported = 0.0
        #         else:
        #             # I feel this is random tie breaking
        #             is_supported = generated_answer.index("true") > generated_answer.index("false")
        #             is_supported = 1.0 if is_supported else 0.0
        #     else:
        #         generated_answer = generated_answer.translate(str.maketrans("", "", string.punctuation)).split()
        #         is_supported = all([keyword not in generated_answer for keyword in ["not", "cannot", "unknown", "information"]])
        #         is_supported = 1.0 if is_supported else 0.0

        #     return is_supported

        # self._agent = ChatInterface(
        #     model_name=self._model_name,
        #     batch_size=32,
        #     max_tokens=32,
        #     system_message=None,
        #     instruction_prompt=[],
        #     input_example_prompt="".join([
        #         "Answer the question about {topic} based on the given context.\n\n",
        #         "{parsed_passages}\n\n",
        #         "Input: {input} True or False?\nOutput:"
        #     ]),
        #     output_example_prompt="",
        #     input_parser=_parse_input,
        #     output_parser=_parse_output,
        #     base_url=self._base_url,
        #     api_key=self._api_key,
        #     max_concurrency=32,
        # )

        self._llm = ChatOpenAI(
            model_name=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            max_tokens=128,
            # top_p=0.98,
            model_kwargs={"top_p": 0.98},
            temperature=0.0,
        )
        self._runnable_config = RunnableConfig(max_concurrency=32)

        self._agent = FActScoreEvidentialSupportStep().chain_llm(self._llm)

        self._retriever = retriever

    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """ """
        # first convert the input instance into the FActScoreQueryInstance
        # by retrieve from the database
        passages = self._retriever.get_passages(instance.topic, instance.text, 5)

        input_instance = {
            "topic": instance.topic,
            "parsed_passages": "\n\n".join(
                [
                    f"Title: {passage['title']} Text: {passage['text']}"
                    for passage in passages
                ]
            )
            + "\n\n",
            "input": instance.text,
        }

        response = self._agent.invoke(input_instance, config=self._runnable_config)

        return {
                "raw": response.messages,
                "parsed": response.evidential_support,
                "support_input": input_instance['parsed_passages']
            }

    @overrides
    def _batch_score(
        self, instances: List[ScorerInstance]
    ) -> List[Dict[Text, Text | float]]:
        """Now we will first retrieve for all the instances."""

        topics = [instance.topic for instance in instances]
        texts = [instance.text for instance in instances]

        passage_chunks = self._retriever.get_passages_batched(
            topics=topics, questions=texts, k=5
        )

        assert len(passage_chunks) == len(instances)

        input_instances = [
            {
                "topic": instance.topic,
                "parsed_passages": "\n\n".join(
                    [
                        f"Title: {passage['title']} Text: {passage['text']}"
                        for passage in passages
                    ]
                )
                + "\n\n",
                "input": instance.text,
            }
            for instance, passages in zip(instances, passage_chunks)
        ]

        responses = self._agent.batch(input_instances, config=self._runnable_config)

        returns = []
        for r, i in zip(responses, input_instances):
            returns.append({
                "raw": r.messages,
                "parsed": r.evidential_support,
                "support_input": i['parsed_passages']
            })

        return returns

