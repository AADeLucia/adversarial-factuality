"""
"""
import os
import sys
from abc import ABC, abstractmethod
from typing import Text, List, Tuple, Union

from overrides import overrides
from registrable import Registrable

# Easier for debugging
try:
    from ..utils.instances import ScorerInstance
except ImportError as err:
    current_folder = os.path.dirname(__file__)
    src_folder = os.path.dirname(current_folder)
    sys.path.append(src_folder)
    from utils.instances import ScorerInstance


class Decomposer(ABC, Registrable):
    def __init__(self):
        super().__init__()
        
    def __call__(self, instance: Union[List[ScorerInstance], ScorerInstance]) -> Union[List[List[ScorerInstance]], List[ScorerInstance]]:
        """
        """
        if not isinstance(instance, ScorerInstance):
            return self._batch_decompose(instance)
        return self._decompose(instance)
        
    @abstractmethod
    def _decompose(self, instance: ScorerInstance) -> List[ScorerInstance]:
        raise NotImplementedError("Override the decomposition to get proper decomposition.")
    
    def _batch_decompose(self, instances: List[ScorerInstance]) -> List[List[ScorerInstance]]:
        """We batch whatever way we like, and try to generate a list of decomposed items,
        with indices indicating from which instance the decomposed instance comes from.
        """
        
        results = []
        
        for instance in instances:
            decomposed = self._decompose(instance)
            results.append(decomposed)
                
        return results