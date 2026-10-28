from dataclasses import dataclass
from typing import Protocol

from positions import PositionCrop

@dataclass(frozen=True)
class Observation:
    visual_state: str
    score: float | None = None
    features: dict | None = None
    reason: str = ""

class Inspector(Protocol)
    def inspect(self, crop: PositionCrop) -> Observation:
        ...

class MockInspector:
    def inspect(self, crop: PositionCrop) -> Observation:
        return Observation(visual_state="pass", score=0.0, reason="mock_prediction")