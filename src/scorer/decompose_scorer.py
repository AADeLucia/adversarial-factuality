"""This type of factuality scorer willl take a summary
(or other knowledge intensive paragraph) and score the
output of this summary by decomposing the summary into
individual facts and scoring each fact individually.
"""

from registrable import Lazy
from dataclasses import dataclass, asdict
import multiprocessing
import logging
from multiprocessing import Process, Pipe
from ..abstention_detector.abstention_detector import AbstentionDetector
from ..aggregator.aggregator import Aggregator
from ..decomposer.decomposer import Decomposer
from ..utils.instances import ScorerInstance, DedupScorerInstance
from typing import Text, Tuple, List, Dict, Union, Any
from overrides import overrides
from .scorer import Scorer


logger = logging.getLogger(__name__)


def _score_claims(
    instance_tuples: List[Tuple[int, DedupScorerInstance]],
    base_scorer_params: Dict[Text, Any],
    score_sender: multiprocessing.connection.Connection
):
    """Use the base scorer to score the instances.
    """
    instances = [dt[1] for dt in instance_tuples]
    base_scorer = Scorer.from_params(base_scorer_params)
    score_dicts: List[Dict[Text, Text | float]] = base_scorer(instances, return_raw=True)
    
    score_sender.send(score_dicts)
    score_sender.close()

def _deduplicate_claims(
    instance_tuples: List[Tuple[int, DedupScorerInstance]],
    decomposer_params: Dict[Text, Any],
    dedup_sender: multiprocessing.connection.Connection
):
    """Use the deduplication process to generate a deduplication selection process.
    """
    
    decomposer = Decomposer.from_params(decomposer_params)
    selected_indices: List[int] = decomposer._batch_deduplicate(instance_tuples)
    dedup_sender.send(selected_indices)
    dedup_sender.close()


@Scorer.register("decompose")
class DecomposeScorer(Scorer):
    
    __NAME__ = "decompose"

    def __init__(
        self,
        abstention_detector: AbstentionDetector,
        decomposer: Decomposer,
        base_scorer: Scorer,
        aggregator: Aggregator,
        with_parallel: bool = False
    ):
        super().__init__()
        self.abstention_detector = abstention_detector
        self.decomposer = decomposer
        self.base_scorer = base_scorer
        self.aggregator = aggregator
        self._with_parallel = with_parallel
        
    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """
        """
        
        if self.abstention_detector(instance.text):
            return {
                "parsed": 0.0,
                "raw": f"[[Abstained Response Detected]]: {instance.text}"
            }
        
        decomposed_instances: List[ScorerInstance] = self.decomposer(instance)
        scores = [self.base_scorer(dt, return_raw=True) for dt in decomposed_instances]

        parsed_scores = [s['parsed'] for s in scores]

        agg_score = self.aggregator(parsed_scores)
        
        return {
            "parsed": agg_score,
            "raw": " ## ".join([str(s['raw']) for s in scores]),
        }
    
    @overrides
    def _batch_score(self, instances: List[ScorerInstance]) -> List[Dict[Text, Text | float]]:
        """We casecade the first step, and use multi-processing to parallelize
        the decomposer and base_scorer.
        """
        
        # if not isinstance(self.decomposer, DeduplicatedDecomposer):
        if not self._with_parallel:
            return self._legacy_batch_score(instances)
        elif self.decomposer.__NAME__ != "deduplicated":
            logger.warning("Decomposer is not a DeduplicatedDecomposer, falling back to legacy batch score.")
            return self._legacy_batch_score(instances)

        # we unpack the decomposer call, first preprocess to prepare for multi-processing
        return self._base_decomposer_unroll_batch_score(instances)
        
    def _base_decomposer_unroll_batch_score(self, instances: List[ScorerInstance]) -> List[Dict[Text, Text | float]]:
        """To successfully run this function, the decomposer must be a DeduplicatedDecomposer,
        because in this case the decomposer will return a list of tuples, where there is
        a base-decomposer to be called before all other actions take place.
        """
        
        results = {}
        
        instance_needs_process = []

        for idx, instance in enumerate(instances):
            if self.abstention_detector(instance.text):
                results[idx] = {
                    "parsed": 0.0,
                    "raw": f"[[Abstained Response Detected]]: {instance.text}"
                }
                continue
            else:
                instance_needs_process.append(idx)
                
        # now we have a list of instances that need processing
        input_instances = [instances[idx] for idx in instance_needs_process]
        
        # 1. Call the base-decomposer to get the atomic fact sets for each sentence.
        # self.decomposer: "DeduplicatedDecomposer"
        sent_instances, sent_tuples = self.decomposer._batch_prepare(input_instances)
        atomic_facts: List[List[ScorerInstance]] = self.decomposer._base_decomposer(sent_instances)

        # 2. Prepare the deduplicator input
        claim_inputs = []
        for sidx, (stuple, fact_sets) in enumerate(zip(sent_tuples, atomic_facts)):
            claim_inputs.extend([(stuple[0], sidx, aidx, atom) for aidx, atom in enumerate(fact_sets)])
            
        claim_checkworthiness = self.decomposer._claim_level_checkworthy_scorer([c[3] for c in claim_inputs])
        
        # create inputs to deduplication tupled with the instance indices
        ccw_inputs = [
            (
                ci[0],
                DedupScorerInstance(
                    text=ci[3].text,
                    topic=ci[3].topic,
                    in_sent_claim_idx=ci[2],
                    from_sent_idx=ci[1],
                    sent=sent_tuples[ci[1]][2],
                    sent_checkworthy=sent_tuples[ci[1]][1],
                    claim_checkworthy=ccw
                )
            ) for ci, ccw in zip(claim_inputs, claim_checkworthiness)]
        
        # create multiple process, and send deduplication to one
        # process, the scoring of all claims to another process.
        dedup_sender, dedup_recver = Pipe()
        score_sender, score_recver = Pipe()
        
        score_process = Process(target=_score_claims, args=(ccw_inputs, self.base_scorer.hparams, score_sender))
        dedup_process = Process(target=_deduplicate_claims, args=(ccw_inputs, self.decomposer.hparams, dedup_sender))
        
        score_process.start()
        dedup_process.start()
        
        dedup_process.join()
        score_process.join()

        # now we have the deduplicated claims, we can now score them
        dedup_results: List[int] = dedup_recver.recv()
        score_results: List[Dict[Text, Text | float]] = score_recver.recv()
        
        # now do the aggregation as unrolled in the _batch_deduplicate
        # as well as in the _batch_score
        deduplicated: List[Tuple[int, DedupScorerInstance]] = [ccw_inputs[idx] for idx in dedup_results]
        score_results: List[Dict[Text, Text | float]] = [score_results[idx] for idx in dedup_results]

        # grouped parsed scores by index
        grouped_parsed_scores = {idx: [] for idx in range(len(instances))}
        for (idx, di), score_dict in zip(deduplicated, score_results):
            grouped_parsed_scores[idx].append({**score_dict, **asdict(di)})

        for idx, score_dicts in grouped_parsed_scores.items():
            if idx not in instance_needs_process:
                assert idx in results, f"Index not processed hasn't been correctly inserted: {idx}."
                continue
            agg_score = self.aggregator([s['parsed'] for s in score_dicts])
            assert idx not in results, f"Index already exists in results {idx}."
            results[idx] = {
                "parsed": agg_score,
                "raw": " ## ".join([str(s['raw']) for s in score_dicts]),
                "claims": [{'text': s['text'], 'score': s['parsed'], 'raw': str(s['raw']), 'support': s.get('support_input')} for s in score_dicts],
            }
            
        return [results[index] for index in range(len(instances))]
    
    def _legacy_batch_score(self, instances: List[ScorerInstance]) -> List[Dict[Text, Text | float]]:
        """We cascade over all the components instead of running components one by one
        on instance.
        """

        results = {}
        
        instance_needs_process = []

        for idx, instance in enumerate(instances):
            if self.abstention_detector(instance.text):
                results[idx] = {
                    "parsed": 0.0,
                    "raw": f"[[Abstained Response Detected]]: {instance.text}"
                }
                continue
            else:
                instance_needs_process.append(idx)
                
        # now we have a list of instances that need processing
        input_instances = [instances[idx] for idx in instance_needs_process]
        decomposed_instance_chunks: List[List[ScorerInstance]] = self.decomposer(input_instances)
        
        # extend them to tuples with index
        decomposed_instance_tuples = [(idx, dt) for idx, dts in zip(instance_needs_process, decomposed_instance_chunks) for dt in dts]
        raw_scores = self.base_scorer([dt[1] for dt in decomposed_instance_tuples], return_raw=True)
        # parsed_scores = [s['parsed'] for s in scores]
        
        # print(decomposed_instance_tuples)
        
        # grouped parsed scores by index
        grouped_parsed_scores = {idx: [] for idx in range(len(instances))}
        for (idx, di), score_dict in zip(decomposed_instance_tuples, raw_scores):
            grouped_parsed_scores[idx].append({**score_dict, **asdict(di)})
            
        for idx, score_dicts in grouped_parsed_scores.items():
            if idx not in instance_needs_process:
                assert idx in results, f"Index not processed hasn't been correctly inserted: {idx}."
                continue
            agg_score = self.aggregator([s['parsed'] for s in score_dicts])
            assert idx not in results, f"Index already exists in results {idx}."
            results[idx] = {
                "parsed": agg_score,
                "raw": " ## ".join([str(s['raw']) for s in score_dicts]),
                "claims": [{'text': s['text'], 'claim_sentence': s.get('claim_sentence'), 'score': s['parsed'], 'raw': str(s['raw']), 'support': s.get('support_input')} for s in score_dicts],
            }
            
        return [results[index] for index in range(len(instances))]