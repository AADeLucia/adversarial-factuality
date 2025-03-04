"""Load pre-computed claims"""

import json
import os.path
import sys

from overrides import overrides
from typing import List, Text, Optional, Tuple

import spacy
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from langchain_interface.example_selectors import ConstantExampleSelector

from langchain_interface.steps.decomposition_step import (
    DecompositionStep,
    DecompositionResponse
)

# Easier for debugging
try:
    from ..utils.instances import ScorerInstance
    from .decomposer import Decomposer
except ImportError as err:
    current_folder = os.path.dirname(__file__)
    src_folder = os.path.dirname(current_folder)
    sys.path.append(src_folder)
    from utils.instances import ScorerInstance
    from decomposer import Decomposer


@Decomposer.register("precomputed_decomposer")
class PrecomputedDecomposer(Decomposer):
    __NAME__ = "precomputed_decomposer"

    def __init__(
            self,
            claims_path: Text,
            topic_key: Text = "id",
            claims_key: Text = "claims",
            nlp_model_name: Text = "en_core_web_sm",
            sentencize: bool = True,
    ):
        """This decomposer expects claims in a JSONlines file
        in {'topic': '', 'claims': []} format
        """
        super().__init__()
        self._nlp = spacy.load(nlp_model_name)  # disable=["ner", "parser"]
        self._nlp.add_pipe("sentencizer")
        self._sentencize = sentencize
        self.precomputed_claims = self._load_claims(claims_path, topic_key, claims_key)

    @overrides
    def _decompose(self, instance: ScorerInstance) -> List[ScorerInstance]:
        """Decompose claims from an instance"""
        output = []
        instance_claims_all = self.precomputed_claims.get(instance.topic, [])
        if self._sentencize:
            instance_sentences = [s for s in self._nlp(instance.text).sents]
            for idx, sentence in enumerate(instance_sentences):
                sentence = sentence.text.strip()
                sentence_claims = instance_claims_all[idx]
                for claim in sentence_claims:
                    output.append(
                        ScorerInstance(
                            text=claim,
                            topic=instance.topic,
                            source_text=instance.source_text,
                            sentence=sentence
                        )
                    )
        else:
            for claim in instance_claims_all:
                output.append(
                    ScorerInstance(
                        text=claim,
                        topic=instance.topic,
                        source_text=instance.source_text,
                        sentence=instance.source_text
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

    def _load_claims(self, claims_path: Text, topic_key: Text, claims_key: Text):
        if not os.path.exists(claims_path):
            raise FileNotFoundError(claims_path)
        claims_by_topic = {}
        with open(claims_path) as claims_file:
            for line in claims_file.readlines():
                row = json.loads(line)
                cleaned_claims = []
                for claim in row.get(claims_key, []):
                    if self._sentencize:
                        assert isinstance(claim, list), "Claims must be a list of lists for proper claim-sentence pairing"
                    # Read list for sentence-level claims
                    if isinstance(claim, list):
                        sentence_claims = [self._clean_claim(c) for c in claim]
                        if not self._sentencize:
                            cleaned_claims.extend(sentence_claims)
                        else:
                            cleaned_claims.append(sentence_claims)
                    # or claims are on the whole text-level
                    # throw error if senticize is True
                    else:
                        claim = claim.replace("-", "").strip()
                        cleaned_claims.append(claim)
                claims_by_topic[row[topic_key]] = cleaned_claims
        return claims_by_topic

    @staticmethod
    def _clean_claim(claim: Text) -> Optional[Text]:
        # Remove "- " in front
        claim = claim.replace("-", "").strip()
        # # Remove "No verifiable claim"
        # if claim == "No verifiable claim":
        #     return None
        return claim


if __name__ == "__main__":
    decomposer = PrecomputedDecomposer(
        claims_path = "/home/hhuan134/scr4_mdredze1/hhuan134/MEDIC/GPT/results/augment_response/fixed_doctor_responses_medscore-decontext_gpt.jsonl",
        topic_key = "id",
        claims_key = "claims",
        sentencize = True,
    )

    example = ScorerInstance(
        topic="yas6ff_20241101",
        text="I spoke to your doctor and they wanted to address your concerns about your irregular periods and extreme pain. They believe that your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods. \n\nYour doctor also mentioned that it's not uncommon for women to experience irregular cycles after stopping birth control, and it may take some time for your body to regulate itself. Additionally, they noted that your weight may be a contributing factor to your symptoms, as excess fat tissue can disrupt hormone levels in the body.\n\nTheir recommendations for you are to consider alternative birth control options that may be more suitable for you, as well as taking over-the-counter pain medications such as NSAIDs to help manage your cramps. They also suggested that losing weight may help alleviate some of your symptoms.\n\nRegarding your concerns about PCOS, your doctor believes that it's unlikely given that multiple scans have shown no cysts on your ovaries. However, they did mention that there could be other underlying causes for your symptoms, and they think that anovulatory cycles and primary dysmenorrhea are more likely explanations for what you're experiencing.",
        source_text="I spoke to your doctor and they wanted to address your concerns about your irregular periods and extreme pain. They believe that your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods. \n\nYour doctor also mentioned that it's not uncommon for women to experience irregular cycles after stopping birth control, and it may take some time for your body to regulate itself. Additionally, they noted that your weight may be a contributing factor to your symptoms, as excess fat tissue can disrupt hormone levels in the body.\n\nTheir recommendations for you are to consider alternative birth control options that may be more suitable for you, as well as taking over-the-counter pain medications such as NSAIDs to help manage your cramps. They also suggested that losing weight may help alleviate some of your symptoms.\n\nRegarding your concerns about PCOS, your doctor believes that it's unlikely given that multiple scans have shown no cysts on your ovaries. However, they did mention that there could be other underlying causes for your symptoms, and they think that anovulatory cycles and primary dysmenorrhea are more likely explanations for what you're experiencing.",
        sentence=None
    )
    o = decomposer._decompose(example)
    for o_i in o:
        print(o_i, "\n\n")
