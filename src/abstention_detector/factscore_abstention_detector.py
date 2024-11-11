"""
"""

from overrides import overrides
import re
import numpy as np
from typing import Text
from .abstention_detector import AbstentionDetector


@AbstentionDetector.register("factscore")
class FActScoreAbstentionDetector(AbstentionDetector):
    def __init__(self):
        super().__init__()
        
        self._invalid_ppl_mentions = [
            "I could not find any information",
            "The search results do not provide",
            "There is no information available",
            "There are no provided search results",
            "not provided in the search results",
            "is not mentioned in the provided search results",
            "There seems to be a mistake in the question",
            "Not sources found",
            "No sources found",
            "Try a more general question"
        ]
        
    @overrides
    def _detect_abstention(self, response: Text) -> bool:
        """
        """
        def remove_citation(text):
            # text = re.sub(r'\[\d+\]', '', text)
            text = re.sub(r"\s*\[\d+\]\s*","", text)
            if text.startswith("According to , "):
                text = text.replace("According to , ", "According to the search results, ")
            return text
        
        def is_invalid_ppl(text):
            return np.any([text.lower().startswith(mention.lower()) for mention in self._invalid_ppl_mentions])
        
        def is_invalid_paragraph_ppl(text):
            return len(text.strip()) == 0 or np.any([mention.lower() in text.lower() for mention in self._invalid_ppl_mentions])
        
        def perplexity_ai_abstain_detect(generation):
            output = remove_citation(generation)
            if is_invalid_ppl(output):
                return True
            
            valid_paras = []

            for para in output.split("\n\n"):
                if is_invalid_paragraph_ppl(para):
                    break
                valid_paras.append(para.strip())
                
            return len(valid_paras) == 0
        
        def generic_abstain_detect(generation):
            return generation.startswith("I'm sorry") or "provide more" in generation
        
        # However, in the original implementation from FActScore, only one of the two abstention detection methods is used.
        # However, since we do work with local models like mistral, Llama, we'll use both methods for stronger abstension.
        # Afterall, this does not need to be applied for our metric, thus should not be a serious issue.
        return perplexity_ai_abstain_detect(response) or generic_abstain_detect(response)