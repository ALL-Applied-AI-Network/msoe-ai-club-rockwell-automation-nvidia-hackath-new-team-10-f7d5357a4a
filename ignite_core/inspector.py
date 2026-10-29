from dataclasses import dataclass
from typing import Protocol

from positions import PositionCrop

@dataclass(frozen=True)
class Observation:
    visual_state: str
    score: float | None = None
    features: dict | None = None
    reason: str = ""

class Inspector(Protocol):
    def inspect(self, crop: PositionCrop) -> Observation:
        ...

class MockInspector:
    def inspect(self, crop: PositionCrop) -> Observation:
        # A placeholder has no visual evidence and must never manufacture PASS.
        return Observation(visual_state="unknown", score=None, reason="mock_no_evidence")
