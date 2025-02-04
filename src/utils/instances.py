"""
"""

from dataclasses import dataclass, field
from typing import Text, Union, Optional, List
from langchain_interface.instances import Instance


@dataclass(frozen=True, eq=True)
class InContextExample(Instance):
    input_text: Text
    generation: Text


@dataclass(frozen=True, eq=True)
class ScorerInstance(Instance):
    text: Text
    source_text: Union[None, Text]
    topic: Union[None, Text]
    sentence: Union[None, Text]


@dataclass(frozen=True, eq=True)
class DedupScorerInstance(ScorerInstance):
    in_sent_claim_idx: int
    from_sent_idx: int
    sent: Text
    sent_checkworthy: float
    claim_checkworthy: float
    
    
@dataclass(frozen=True, eq=True)
class DecontextScorerInstance(ScorerInstance):
    sent: Text


@dataclass(frozen=True, eq=True)
class RelevancyScorerInstance(ScorerInstance):
    sent: Text


@dataclass(frozen=True, eq=True)
class LLMQueryInstance(Instance):
    id: Optional[int] = None
    input: Optional[Text] = None
    output: Optional[Text] = None
    
    
@dataclass(frozen=True, eq=True)
class DecontextInstance(LLMQueryInstance):
    sentence: Optional[Text] = ""


@dataclass(frozen=True, eq=True)
class RelevancyInstance(LLMQueryInstance):
    question: Optional[Text] = ""
    sentence: Optional[Text] = ""


@dataclass(frozen=True, eq=True)
class NextSearchInstance(LLMQueryInstance):
    knowledge: List[Text] = field(default_factory=list)