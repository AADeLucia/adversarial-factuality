""""""

import json
import os
from tqdm import tqdm
from overrides import overrides
from typing import Text
from .task import Task
from src.scorer.scorer import Scorer
from src.utils.instances import ScorerInstance


@Task.register("score-generation-reference")
class ScoreGenerationReferenceTask(Task):
    """
    """
    ___NAME__ = "score-generation-reference"
    
    def __init__(
        self,
        scorer: Scorer,
        input_path: Text,
        text_key: Text,
        topic_key: Text,
        source_key: Text,
        output_path: Text
    ):
        super().__init__()
        self.scorer = scorer
        self.input_path = input_path
        self.output_path = output_path
        self.topic_key = topic_key
        self.source_key = source_key
        self.text_key = text_key
        
    @overrides
    def run(self):
        """
        """
        # Load file in JSON format
        # Check if JSON-lines or JSON
        if self.input_path.endswith(".jsonl"):
            with open(self.input_path, "r", encoding='utf-8') as file_:
                data = [json.loads(line) for line in file_]
        else:
            with open(self.input_path, "r", encoding='utf-8') as file_:
                data = json.load(file_)

        # Prepare input
        inputs = [ScorerInstance(text=item[self.text_key], topic=item[self.topic_key], source_text=item[self.source_key]) for item in tqdm(data)]

        # Run
        results = self.scorer(inputs, return_raw=True)
        results = [{**item, "score": result['parsed'], 'meta': result} for result, item in zip(results, data)]

        # Save
        with open(self.output_path, "w", encoding='utf-8') as file_:
            for result in results:
                file_.write(json.dumps(result) + "\n")

