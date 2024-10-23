""" Mostly follows factscore_decomposer (but apply filterings before and after) """

import spacy
import numpy as np
from dataclasses import dataclass
from overrides import overrides
from typing import List, Text, Tuple, Dict, Any, Union
from ..entailer.entailer import Entailer, EntailerInstance
from scipy.optimize import milp, Bounds, LinearConstraint
from langchain_interface.example_selectors import ConstantExampleSelector
from ..scorer.scorer import Scorer
from ..utils.instances import ScorerInstance, DedupScorerInstance
from .decomposer import Decomposer


@Decomposer.register("deduplicated")
class DeduplicatedDecomposer(Decomposer):
    """Deduplicate the facts from the base decomposer."""

    __NAME__ = "deduplicated"

    def __init__(
        self,
        base_decomposer: Decomposer,
        sentence_level_checkworthy_scorer: Scorer,
        claim_level_checkworthy_scorer: Scorer,
        entailer: Entailer,
        sentencize: bool = True,
    ):
        """ """
        super().__init__()
        self._base_decomposer = base_decomposer
        self._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self._nlp.add_pipe("sentencizer")
        self._sentencize = sentencize

        self._sentence_level_checkworthy_scorer = sentence_level_checkworthy_scorer
        self._claim_level_checkworthy_scorer = claim_level_checkworthy_scorer
        self._entailer = entailer

    def _decompose(self, instance: ScorerInstance) -> List[ScorerInstance]:
        """ """

        all_instances = []

        # filter out duplicated claims (on sentence-level)
        filtered_decomposed = []
        claim_to_sent: Dict[Text, List[Dict[Text, Any]]] = {}
        extant_claims = set()

        if self._sentencize:
            sent_seqs = [
                ScorerInstance(text=sent, topic=instance.topic, source_text=instance.source_text)
                for sent in self._to_sents(instance.text)
            ]
        else:
            sent_seqs = [instance]

        checkworthiness = self._sentence_level_checkworthy_scorer(
            sent_seqs, return_raw=True
        )
        checkworthiness = [c["parsed"] for c in checkworthiness]

        for sidx, (sent, sent_checkworthy) in enumerate(
            zip(sent_seqs, checkworthiness)
        ):
            # TODO: optimize the threshold selection
            if sent_checkworthy < 0.5:
                # This sentence is not checkworthy in general
                continue
            decomposed: List[ScorerInstance] = self._base_decomposer(sent)

            # filter out duplicated claims (on sentence-level)
            # filtered_decomposed = []
            # extant_claims = set()

            for didx, decomposed_instance in enumerate(decomposed):
                if decomposed_instance.text not in extant_claims:
                    extant_claims.add(decomposed_instance.text)
                    filtered_decomposed.append(decomposed_instance)
                if decomposed_instance.text not in claim_to_sent:
                    claim_to_sent[decomposed_instance.text] = []
                claim_to_sent[decomposed_instance.text].append(
                    {
                        "from_sent_idx": sidx,
                        "in_sent_claim_idx": didx,
                        "sent": sent.text,
                    }
                )

        claim_checkworthiness = self._claim_level_checkworthy_scorer(
            filtered_decomposed
        )
        # print("s:", claim_checkworthiness)

        all_instances = [
            DedupScorerInstance(
                text=decomposed_instance.text,
                topic=decomposed_instance.topic,
                source_text=decomposed_instance.source_text,
                # in_sent_claim_idx=claim_idx,
                # from_sent_idx=idx,
                # sent=sent,
                sent_checkworthy=sent_checkworthy,
                claim_checkworthy=claim_checkworthy,
                **sent_dict
            )
            for decomposed_instance, claim_checkworthy in zip(
                filtered_decomposed, claim_checkworthiness
            )
            for sent_dict in claim_to_sent[decomposed_instance.text]
        ]

        # print("s:", [i.in_sent_claim_idx for i in all_instances])

        # we already get all the instances from the base decomposer,
        # now we will run the deduplication

        if not all_instances:
            return []

        selection = self._deduplicate(all_instances)
        return [all_instances[index] for index in selection]

    @overrides
    def _batch_decompose(
        self, instances: List[ScorerInstance]
    ) -> List[List[ScorerInstance]]:
        """ """

        sent_instances, sent_tuples = self._batch_prepare(instances)
        atomic_facts = self._base_decomposer(sent_instances)

        # group atomic facts based on  sentence origins
        claim_inputs = []
        for sidx, (stuple, fact_sets) in enumerate(zip(sent_tuples, atomic_facts)):
            claim_inputs.extend(
                [(stuple[0], sidx, aidx, atom) for aidx, atom in enumerate(fact_sets)]
            )

        claim_checkworthiness = self._claim_level_checkworthy_scorer(
            [c[3] for c in claim_inputs]
        )
        # print("b:", claim_checkworthiness)

        # create inputs to deduplication tupled with the instance indices
        ccw_inputs = [
            (
                ci[0],
                DedupScorerInstance(
                    text=ci[3].text,
                    topic=ci[3].topic,
                    source_text=ci[3].source_text,
                    in_sent_claim_idx=ci[2],
                    from_sent_idx=ci[1],
                    sent=sent_tuples[ci[1]][2],
                    sent_checkworthy=sent_tuples[ci[1]][1],
                    claim_checkworthy=ccw,
                ),
            )
            for ci, ccw in zip(claim_inputs, claim_checkworthiness)
        ]

        # print("b:", [i.in_sent_claim_idx for _, i in ccw_inputs])

        selection_index = self._batch_deduplicate(ccw_inputs)
        deduplicated = [ccw_inputs[idx] for idx in selection_index]

        # group them by the original instance index
        grouped_deduplicated = {}
        for idx, instance in deduplicated:
            if idx not in grouped_deduplicated:
                grouped_deduplicated[idx] = []
            grouped_deduplicated[idx].append(instance)

        return [
            grouped_deduplicated[idx] if idx in grouped_deduplicated else []
            for idx in range(len(instances))
        ]

    def _batch_prepare(
        self, instances: List[ScorerInstance]
    ) -> Tuple[List[ScorerInstance], List[Tuple[int, Text]]]:
        """Notice that currently there's no need to implement a non-batched
        version of this function.
        """
        if self._sentencize:
            # need to first breakdown into sentences
            sent_tuples = [
                (idx, sent)
                for idx, instance in enumerate(instances)
                for sent in self._to_sents(instance.text)
            ]
        else:
            sent_tuples = [
                (idx, instance.text) for idx, instance in enumerate(instances)
            ]

        sent_instances = [
            ScorerInstance(text=sent, topic=instances[idx].topic, source_text=instances[idx].source_text)
            for idx, sent in sent_tuples
        ]

        checkworthiness = self._sentence_level_checkworthy_scorer(
            sent_instances, return_raw=True
        )
        checkworthiness = [c["parsed"] for c in checkworthiness]

        # filter down to those checkworthy
        sent_tuples = [
            (idx, ckwt, sent)
            for (idx, sent), ckwt in zip(sent_tuples, checkworthiness)
            if ckwt > 0.5
        ]
        return [
            ScorerInstance(text=sent, topic=instances[idx].topic, source_text=instances[idx].source_text)
            for idx, _, sent in sent_tuples
        ], sent_tuples

    def _batch_deduplicate(
        self, instance_tuples: List[Tuple[int, DedupScorerInstance]]
    ) -> List[int]:
        """
        instance_tuples: --- List[Tuple[int, DedupScorerInstance]],
            where the first element of the tuple is the index of the instance the dedup comes from
            and the second element is the DedupScorerInstance.

        return: --- List[int],
            the indices of the instances that are not duplicated (need to be selected).
        """

        # group instances by the first integer from the tuple
        sent_ent_results = self._entailer(
            [
                EntailerInstance(premise=instance.sent, hypothesis=instance.text)
                for _, instance in instance_tuples
            ],
            desc="Premise=Sent, Hypothesis=Claim",
        )
        index_swap = [
            tidx
            for tidx, (tp, er) in enumerate(zip(instance_tuples, sent_ent_results))
            if er > 0.5
        ]

        # grouped_instances: Dict[Text, Dict[Text, Dict[Text, Union[int, float]]]] = {}
        grouped_instances: Dict[Text, List[Dict[Text, Union[int, float]]]] = {}

        for iidx, (er, old_index) in enumerate(zip(
            sent_ent_results, index_swap
        )):
            idx, instance = instance_tuples[old_index]
            if idx not in grouped_instances:
                grouped_instances[idx] = []
            grouped_instances[idx].append(
                {"score": er, "from_index": iidx, "text": instance.text}
            )

        finding_pair_dicts = {}
        pairwise_entailment_inputs = []

        for idx, text_dicts in grouped_instances.items():
            if idx not in finding_pair_dicts:
                finding_pair_dicts[idx] = {}
            for i in range(len(text_dicts)):
                for j in range(len(text_dicts)):
                    if i == j:
                        continue
                    elif text_dicts[i]["text"] == text_dicts[j]["text"]:
                        finding_pair_dicts[idx][(i, j)] = -1  # automatic entailment
                    finding_pair_dicts[idx][(i, j)] = len(pairwise_entailment_inputs)
                    pairwise_entailment_inputs.append(
                        EntailerInstance(
                            premise=text_dicts[i]["text"],
                            hypothesis=text_dicts[j]["text"],
                        )
                    )

        parwise_entailment_scoring = self._entailer(pairwise_entailment_inputs, desc="Pairwise Entailment")

        # create intra_entailment_matrix for each of the group
        intra_entailment_matrices = {}

        for idx, text_dicts in grouped_instances.items():
            intra_entailment_matrices[idx] = np.array(
                [
                    [
                        (
                            (
                                parwise_entailment_scoring[
                                    finding_pair_dicts[idx][(i, j)]
                                ]
                                > 0.5
                                if finding_pair_dicts[idx][(i, j)] >= 0
                                else True
                            )
                            if i != j
                            else False
                        )
                        for j in range(len(text_dicts))
                    ]
                    for i in range(len(text_dicts))
                ],
                dtype=np.int16,
            )

        # solve the MILP problem for each group
        MILP_results = {}

        for idx, text_dicts in grouped_instances.items():
            selection, result = self._solve_milp(
                pairwise_entailment=intra_entailment_matrices[idx],
                # Zhengping 05/26/2024: now we are using the claim checkworthiness matched to the filtered instances
                weighting=np.array(
                    [
                        instance_tuples[index_swap[tdict["from_index"]]][1].sent_checkworthy
                        * instance_tuples[index_swap[tdict["from_index"]]][1].claim_checkworthy
                        for tdict in text_dicts
                    ],
                    np.float32,
                ),
            )

            non_zero_selection_indices = np.nonzero(selection)[0].tolist()
            MILP_results[idx] = non_zero_selection_indices

        # fetch back results using MILP_results
        # return_results = []
        # for idx, texts in grouped_texts.items():
        #     return_results.extend([instance_tuples[texts[selected_idx][0]] for selected_idx in MILP_results[idx]])

        # join all MILP_results
        return_ids = set()
        for idx, text_dicts in grouped_instances.items():
            return_ids.update(
                [index_swap[text_dicts[selected_idx]['from_index']] for selected_idx in MILP_results[idx]]
            )

        selection_indices = sorted(return_ids, key=lambda x: x, reverse=False)
        return selection_indices

    def _to_sents(self, text: Text) -> List[Text]:
        """ """

        return [sentence.text for sentence in self._nlp(text).sents]

    def _deduplicate(self, instances: List[DedupScorerInstance]) -> List[int]:
        """Return the indices of the instances that are not duplicated."""

        sent_filter_instances = [
            EntailerInstance(premise=instance.sent, hypothesis=instance.text)
            for instance in instances
        ]
        sent_ent_results = self._entailer(sent_filter_instances, desc="Premise=Sent, Hypothesis=Claim")

        # filter out claims that are not entailed
        instances_wreal_idx = [
            (tidx, instance)
            for tidx, (instance, entailed) in enumerate(
                zip(instances, sent_ent_results)
            )
            if entailed > 0.5
        ]

        # create pairwise entailment instances
        # if not in the result is 1
        finding_pair: Dict[Tuple[int, int], int] = {}
        pairwise_entailment_instances = []

        for i in range(len(instances_wreal_idx)):
            for j in range(len(instances_wreal_idx)):
                if i == j:
                    continue
                elif instances_wreal_idx[i][1].text == instances_wreal_idx[j][1].text:
                    finding_pair[(i, j)] = -1  # automatic entailment
                else:
                    finding_pair[(i, j)] = len(pairwise_entailment_instances)
                    pairwise_entailment_instances.append(
                        EntailerInstance(
                            premise=instances_wreal_idx[i][1].text, hypothesis=instances_wreal_idx[j][1].text
                        )
                    )

        pairwise_entailment_scoring = self._entailer(pairwise_entailment_instances, desc="Pairwise Entailment")

        intra_ent_mat = np.array(
            [
                [
                    (
                        (
                            pairwise_entailment_scoring[finding_pair[(i, j)]] > 0.5
                            if finding_pair[(i, j)] >= 0
                            else True
                        )
                        if i != j
                        else False
                    )
                    for j in range(len(instances_wreal_idx))
                ]
                for i in range(len(instances_wreal_idx))
            ],
            dtype=np.int16,
        )

        # also, create the weighting vector from the instance
        # TODO: check whether this setting is optimal
        weighting = np.array(
            [
                instance.sent_checkworthy * instance.claim_checkworthy
                for _, instance in instances_wreal_idx
            ],
            np.float32,
        )

        selection, result = self._solve_milp(
            pairwise_entailment=intra_ent_mat, weighting=weighting
        )

        non_zero_selection_indices = sorted(np.nonzero(selection)[0].tolist())
        non_zero_selection_indices = [instances_wreal_idx[idx][0] for idx in non_zero_selection_indices]
        return non_zero_selection_indices

    # solve the MILP problem
    def _solve_milp(
        self,
        pairwise_entailment: np.ndarray,
        weighting: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """ """

        if pairwise_entailment.size == 0:
            return np.array([], np.int16), 0.0

        or_mat = np.tril(
            np.bitwise_or(pairwise_entailment, np.transpose(pairwise_entailment)),
        )

        indices = np.nonzero(or_mat)

        # TODO: add logging information

        constraints = np.zeros((len(indices[0]), len(weighting)), dtype=np.float32)

        constraints[np.arange(len(indices[0])), indices[0]] = 1
        constraints[np.arange(len(indices[1])), indices[1]] = 1

        res = milp(
            c=-weighting,
            integrality=np.ones_like(weighting),
            bounds=Bounds(
                lb=np.zeros_like(weighting) - 1e-8, ub=np.ones_like(weighting) + 1e-8
            ),
            constraints=(
                LinearConstraint(
                    A=constraints,
                    ub=np.ones(len(indices[0])) + 1e-8,
                ),
            ),
        )

        selection = res.x
        result = res.fun

        return selection, result
