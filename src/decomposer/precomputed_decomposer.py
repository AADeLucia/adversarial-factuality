"""Load pre-computed claims"""

import json
import os.path

from overrides import overrides
from typing import List, Text, Optional, Tuple

import spacy
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from langchain_interface.example_selectors import ConstantExampleSelector

from ..utils.instances import ScorerInstance
from langchain_interface.steps.decomposition_step import (
    DecompositionStep,
    DecompositionResponse
)
from .decomposer import Decomposer


def _load_claims(claims_path: Text, topic_key: Text, claims_key: Text):
    if not os.path.exists(claims_path):
        raise FileNotFoundError(claims_path)
    claims_by_topic = {}
    with open(claims_path) as claims_file:
        for line in claims_file:
            row = json.loads(line)
            claims_by_topic[row[topic_key]] = row.get(claims_key, [])
    return claims_by_topic


@Decomposer.register("precomputed_decomposer")
class PrecomputedDecomposer(Decomposer):
    __NAME__ = "precomputed_decomposer"

    def __init__(
            self,
            claims_path: Text,
            topic_key: Text = "id",
            claims_key: Text = "claims",
    ):
        """This decomposer expects claims in a JSONlines file
        in {'topic': '', 'claims': []} format
        """
        super().__init__()
        self.precomputed_claims = _load_claims(claims_path, topic_key, claims_key)

    @overrides
    def _decompose(self, instance: ScorerInstance) -> List[ScorerInstance]:
        """Decompose claims from an instance"""
        output = []
        for claim in self.precomputed_claims.get(instance.topic, []):
            output.append(
                ScorerInstance(
                    text=claim,
                    topic=instance.topic,
                    source_text=instance.source_text
                )
            )
        return output

    @overrides
    def _batch_decompose(self, instances: List[ScorerInstance]) -> List[List[ScorerInstance]]:
        """Return decomposition from multiple instances"""
        output = []
        for instance in instances:
            output.append(self._decompose(instance))
        return output

