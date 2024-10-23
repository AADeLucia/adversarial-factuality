""" This score scores a given claim by how much confidence boost stating that claim
explicitly gives to UNLI being able to hypothetically predict the claim. """

from dataclasses import dataclass, field
import string
import ujson as json
from typing import Text, Dict, List, Union, Optional
import numpy as np
from overrides import overrides
from ..entailer.entailer import Entailer, EntailerInstance
from ..utils.instances import ScorerInstance
from .scorer import Scorer


@Scorer.register("unli-confidence-boost")
class UNLIConfidenceBoostScorer(Scorer):
    """1 - p(claim | bleached_context)"""

    __NAME__ = "unli-confidence-boost"

    def __init__(
        self,
        bleached_templates: List[Text],
        entailer: Entailer,
        cap_entailer: Optional[Entailer] = None,
        epsilon: float = 1e-3,
    ):
        """We don't explicitly require the entailer to
        be soft, but practically people should always use a
        soft entailer for proper tie-breaking.
        """

        super().__init__()
        
        self._bleached_templates = bleached_templates
        self._entailer = entailer
        self._cap_entailer = cap_entailer
        self._epsilon = epsilon

    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """Here the scorer will score the instance into
        result dict
        """

        bleached_context = [bt.format(topic=instance.topic) for bt in self._bleached_templates]

        # pair each bleached_context with the claim
        # and score each pair

        inputs = [
            EntailerInstance(premise=bc, hypothesis=instance.text)
            for bc in bleached_context
        ]

        scores = self._entailer(inputs, desc="UNLI Scoring")
        # Adding lowerbound to cap_score to avoid log(0)
        score = max(*scores, self._epsilon)

        cap_entailer_outputs = {}

        if self._cap_entailer is not None:
            # if cap_score == 1, then the claim needs to be
            # capped regardless of the score
            cap_scores = self._cap_entailer(inputs, desc="Cap Scoring")
            cap_score = max(cap_scores)
            score = max(score, cap_score)

            cap_entailer_outputs["cap_scores"] = cap_scores

        # Zhengping 05/24/2025: Use - log(score) to align with CPMI
        parsed_score = (- np.log(score) - self._epsilon).item()

        return {
            "premises": bleached_context,
            "hypothesis": instance.text,
            "parsed": parsed_score,
            "scores": scores,
            **cap_entailer_outputs,
        }

    @overrides
    def _batch_score(
        self, instances: List[ScorerInstance]
    ) -> List[Dict[Text, Text | float]]:
        """Run scores in batch."""
        
        inputs = [
            EntailerInstance(premise=bt.format(topic=instance.topic), hypothesis=instance.text)
            for instance in instances
            for bt in self._bleached_templates
        ]

        all_scores = self._entailer(inputs, desc="UNLI Scoring")
        all_scores = [
            all_scores[
                i * len(self._bleached_templates) : (i + 1) * len(self._bleached_templates)
            ]
            for i in range(len(instances))
        ]

        all_cap_entailer_outputs = [{}] * len(instances)

        if self._cap_entailer is not None:
            cap_scores = self._cap_entailer(inputs, desc="Cap Scoring")
            all_cap_entailer_outputs = [
                {
                    "cap_scores": cap_scores[
                        i
                        * len(self._bleached_templates) : (i + 1)
                        * len(self._bleached_templates)
                    ]
                }
                for i in range(len(instances))
            ]

        return [
            {
                "premises": [bt.format(topic=instance.topic) for bt in self._bleached_templates],
                "hypothesis": instance.text,
                "parsed": (
                    # 1 - self._epsilon - max(scores)
                    - np.log(max(*scores, self._epsilon)) - self._epsilon
                    if "cap_scores" not in cap_entailer_outputs
                    else - np.log(max(*scores, *cap_entailer_outputs["cap_scores"], self._epsilon)) - self._epsilon
                ).item(),
                "scores": scores,
                **cap_entailer_outputs,
            }
            for instance, scores, cap_entailer_outputs in zip(
                instances, all_scores, all_cap_entailer_outputs
            )
        ]


@Scorer.register("unli-confidence-boost-from-file")
class UNLIConfidenceBoostFromFileScorer(UNLIConfidenceBoostScorer):
    """1 - p(claim | bleached_context)"""

    __NAME__ = "unli-confidence-boost-from-file"

    def __init__(
            self,
            bleached_templates_path: Text,
            entailer: Entailer,
            cap_entailer: Optional[Entailer] = None,
            epsilon: float = 1e-3,
    ):
        """We don't explicitly require the entailer to
        be soft, but practically people should always use a
        soft entailer for proper tie-breaking.
        """
        with open(bleached_templates_path, "r") as f:
            bleached_templates = f.read().splitlines()

        super().__init__(
            bleached_templates,
            entailer,
            cap_entailer,
            epsilon
        )


@Scorer.register("unli-confidence-boost-from-question")
class UNLIConfidenceBoostFromQuestionScorer(Scorer):
    """1 - p(claim | bleached_context)"""

    __NAME__ = "unli-confidence-boost"

    def __init__(
        self,
        bleach_template_path: Text,
        num_bleached_templates: int,
        entailer: Entailer,
        cap_entailer: Optional[Entailer] = None,
        epsilon: float = 1e-3,
    ):
        """We don't explicitly require the entailer to
        be soft, but practically people should always use a
        soft entailer for proper tie-breaking.
        
        Different from UNLIConfidenceBoostScorer, this scorer use topic
        dependent bleached context
        """

        super().__init__()
        
        self._entailer = entailer
        self._cap_entailer = cap_entailer
        self._epsilon = epsilon
        
        self._bleach_template_path = bleach_template_path
        self._num_bleached_templates = num_bleached_templates

        with open(bleach_template_path, "r", encoding="utf-8") as file_:
            self._bleached_template_dict = json.load(file_)
            
    def _prepare_bleached_context(self, topic: Text) -> List[Text]:
        bleached_context_list_for_topic = self._bleached_template_dict[topic][:self._num_bleached_templates]
        assert bleached_context_list_for_topic, f"bleached context for topic {topic} is empty."
        if len(bleached_context_list_for_topic) < self._num_bleached_templates:
            bleached_context_list_for_topic = bleached_context_list_for_topic + [bleached_context_list_for_topic[-1]] * (self._num_bleached_templates - len(bleached_context_list_for_topic))
            
        return bleached_context_list_for_topic
        
    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """Here the scorer will score the instance into
        result dict
        """

        # bleached_context = [bt.format(topic=instance.topic) for bt in self._bleached_templates]
        bleached_context = self._prepare_bleached_context(instance.topic)

        # pair each bleached_context with the claim
        # and score each pair

        inputs = [
            EntailerInstance(premise=bc, hypothesis=instance.text)
            for bc in bleached_context
        ]

        scores = self._entailer(inputs, desc="UNLI Scoring")
        # Adding lowerbound to cap_score to avoid log(0)
        score = max(*scores, self._epsilon)

        cap_entailer_outputs = {}

        if self._cap_entailer is not None:
            # if cap_score == 1, then the claim needs to be
            # capped regardless of the score
            cap_scores = self._cap_entailer(inputs, desc="Cap Scoring")
            cap_score = max(cap_scores)
            score = max(score, cap_score)

            cap_entailer_outputs["cap_scores"] = cap_scores

        # Zhengping 05/24/2025: Use - log(score) to align with CPMI
        parsed_score = (- np.log(score) - self._epsilon).item()

        return {
            "premises": bleached_context,
            "hypothesis": instance.text,
            "parsed": parsed_score,
            "scores": scores,
            **cap_entailer_outputs,
        }

    @overrides
    def _batch_score(
        self, instances: List[ScorerInstance]
    ) -> List[Dict[Text, Text | float]]:
        """Run scores in batch."""
        
        bleached_context_dict = {instance.topic: self._prepare_bleached_context(instance.topic) for instance in instances}
        
        inputs = [
            EntailerInstance(premise=bt, hypothesis=instance.text)
            for instance in instances
            for bt in bleached_context_dict[instance.topic]
        ]

        all_scores = self._entailer(inputs, desc="UNLI Scoring")
        all_scores = [
            all_scores[
                i * self._num_bleached_templates : (i + 1) * self._num_bleached_templates
            ]
            for i in range(len(instances))
        ]

        all_cap_entailer_outputs = [{}] * len(instances)

        if self._cap_entailer is not None:
            cap_scores = self._cap_entailer(inputs, desc="Cap Scoring")
            all_cap_entailer_outputs = [
                {
                    "cap_scores": cap_scores[
                        i
                        * self._num_bleached_templates : (i + 1)
                        * self._num_bleached_templates
                    ]
                }
                for i in range(len(instances))
            ]

        return [
            {
                "premises": bleached_context_dict[instance.topic],
                "hypothesis": instance.text,
                "parsed": (
                    # 1 - self._epsilon - max(scores)
                    - np.log(max(*scores, self._epsilon)) - self._epsilon
                    if "cap_scores" not in cap_entailer_outputs
                    else - np.log(max(*scores, *cap_entailer_outputs["cap_scores"], self._epsilon)) - self._epsilon
                ).item(),
                "scores": scores,
                **cap_entailer_outputs,
            }
            for instance, scores, cap_entailer_outputs in zip(
                instances, all_scores, all_cap_entailer_outputs
            )
        ]