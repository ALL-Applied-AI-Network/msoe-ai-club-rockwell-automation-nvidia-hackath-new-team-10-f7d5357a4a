from dataclasses import dataclass
from enum import Enum


class FinalState(str, Enum):
    PASS = "PASS"
    DEFECT = "DEFECT"
    MISSED = "MISSED"
    UNSURE = "UNSURE"
    WAITING = "WAITING"


EXPECTED_POSITIONS = tuple(f"G{i:02d}" for i in range(1, 31))


@dataclass(frozen=True, slots=True)
class HoleResult:
    position: str
    predicted_state: str
    final_state: FinalState
    score: float | None
    reason: str
    cap_fraction: float | None
    cap_offset: float | None


@dataclass(frozen=True, slots=True)
class EventSnapshot:
    run: str
    event: str
    frame: str
    fill_frontier: int | None
    holes: tuple[HoleResult, ...]

    def __post_init__(self):
        positions = tuple(h.position for h in self.holes)
        if positions != EXPECTED_POSITIONS:
            raise ValueError("event must contain ordered G01-G30")