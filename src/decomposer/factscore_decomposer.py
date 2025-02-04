"""Replicate the FactScore Decomposer
that takes in a summary and decomposes it into
atomic facts.
"""

import spacy
import json
from overrides import overrides
from typing import List, Text, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from ..utils.instances import ScorerInstance
from ..langchain_step.decomposition_step import (
    DecompositionStep,
    DecompositionResponse
)
from .decomposer import Decomposer


@Decomposer.register("factscore")
class FActScoreDecomposer(Decomposer):
    
    __NAME__ = "factscore"
    
    def __init__(
        self,
        model_name: Text,
        prompt_path: Text,
        nlp_model_name: Text = "en_core_web_sm",
        sentencize: bool = True,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
    ):
        """In general, this decomposer runs a sentence splitter,
        and then an atomic fact extractor to get the atomic facts.
        """

        super().__init__()
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        self._nlp = spacy.load(nlp_model_name, disable=["ner", "parser"])
        self._nlp.add_pipe("sentencizer")
        self._sentencize = sentencize
        
        self._llm = ChatOpenAI(
            model_name=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            max_tokens=512,
            top_p=0.98,
            # model_kwargs={"top_p": 0.98},
            temperature=0.0,
        )

        with open(prompt_path, "r") as file_:
            prompt_settings = json.load(file_)

        self._agent = DecompositionStep(
            system_prompt=prompt_settings.get("system_prompt", ""),
            input_example_prompt=prompt_settings.get("input_prompt", ""),
            examples=prompt_settings.get("examples")
        ).chain_llm(self._llm)
        self._runnable_config = RunnableConfig(max_concurrency=32)
        
    @overrides
    def _decompose(self, instance: ScorerInstance) -> List[ScorerInstance]:
        """ """
        instance_text = instance.text
        topic = instance.topic

        inputs = []
        if self._sentencize:
            for sentence in self._nlp(instance_text).sents:
                s = sentence.text.strip()
                if s:
                    inputs.append({"input": s})
        else:
            inputs.append({"input": instance_text.strip()})

        raw_outputs = self._agent.batch(inputs, config=self._runnable_config)
        outputs = []
        for otp in raw_outputs:
            for atom in otp.claims:
                if self._is_valid_claim(atom):
                    outputs.append(ScorerInstance(text=atom, topic=topic, source_text=instance.source_text))

        return outputs

    def _is_valid_claim(self, claim: Text) -> bool:
        claim = claim.lower().strip()
        # if not claim:
        #     return False
        # if "no verifiable claim" in claim:
        #     return False
        # if "need more context" in claim:
        #     return False
        return True

    @overrides
    def _batch_decompose(self, instances: List[ScorerInstance]) -> List[List[ScorerInstance]]:
        """
        """
        if not self._sentencize:
            inputs = [{"input": instance.text.strip()} for idx, instance in enumerate(instances)]
            num_sents = [1 for _ in instances]
        else:
            inputs = []
            num_sents = []
            
            for idx, instance in enumerate(instances):
                sents = list(self._nlp(instance.text).sents)
                num_sents.append(len(sents))
                for sentence in sents:
                    inputs.append({"input": sentence.text.strip()})
        outputs = self._agent.batch(inputs, config=self._runnable_config)
        
        # now since we are getting all outputs
        results = []
        inputs_idx = 0
        for nidx, ns in enumerate(num_sents):
            if ns == 0:
                results.append([])
            else:
                slice_ = outputs[:ns]
                res = []
                for otp in slice_:
                    orig_sentence = inputs[inputs_idx]["input"]
                    for atom in otp.claims:
                        res.append(
                            ScorerInstance(text=atom, topic=instances[nidx].topic, source_text=instances[nidx].source_text, sentence=orig_sentence)
                        )
                    inputs_idx += 1
                results.append(res)
                outputs = outputs[ns:]

        assert len(outputs) == 0, "Outputs should be empty at the end of the loop."
        return results