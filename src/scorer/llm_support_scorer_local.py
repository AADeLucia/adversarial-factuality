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

@Scorer.register("llm-support-local")
class LLMSupportLocalScorer(Scorer):

    __NAME__ = "llm-support-local"

    def __init__(
        self,
        model_name: Text,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
    ):
        """ """
        super().__init__()

        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        self._llm = ChatOpenAI(
            model_name=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            max_tokens=128,
            model_kwargs={"top_p": 0.98},
            temperature=0.0,
        )
        self._runnable_config = RunnableConfig(max_concurrency=32)
        self._agent = FActScoreEvidentialSupportStep().chain_llm(self._llm)

    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """Score an instance based on source/reference"""
        input_instance = {
            "parsed_passages": f"{instance.source_text}\n\n",
            "input": instance.text,
        }

        response = self._agent.invoke(input_instance, config=self._runnable_config)
        return {"raw": response.messages, "parsed": response.evidential_support, "support_input": instance.source_text}

    @overrides
    def _batch_score(
        self, instances: List[ScorerInstance]
    ) -> List[Dict[Text, Text | float]]:
        """Now we will first retrieve for all the instances."""
        input_instances = [
            {
                "topic": instance.topic,
                "parsed_passages": f"{instance.source_text}\n\n",
                "input": instance.text,
            }
            for instance in instances
        ]

        responses = self._agent.batch(input_instances, config=self._runnable_config)
        results = []
        for response, input_instance in zip(responses, instances):
            results.append({
                "raw": response.messages,
                "parsed": response.evidential_support,
                "support_input": input_instance.source_text,
            })

        return results

