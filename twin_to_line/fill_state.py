"""Fill-sequence reasoning (D007, D021): turn per-position VISUAL observations into PROCESS states.

Kept separate from vision on purpose: the classifier only says what a crop looks like; this module knows the fill
order (G01..G30 = twin numbering = robot fill order; our own convention, not CSI's numbering, D022).

Visual states (same terms as eval_metrics.py): pass | missing_cap | crooked_cap | damaged_vial | no_vial | unknown
Process states: PASS | DEFECT | MISSED | WAITING | UNSURE

The manufacturing equipment provides NO placement count and NO order-complete signal (organizers, Oct 16), so the
runtime decides from occupancy. F = the highest position that visibly holds a vial (the fill frontier).
Two operating contexts (`mode`):
  LIVE_FILL (default)  the fixture may still be filling.
    occupied           : pass -> PASS, cap/vial defect -> DEFECT
    unknown visual     : UNSURE (occluded, ambiguous)
    empty, i < F       : MISSED  (a later position is filled, so production has passed it)
    empty, i == F+1    : frontier policy, WAITING (default) or UNSURE: not attempted yet vs just skipped
    empty, i > F+1     : WAITING (never MISSED while filling may continue)
  FINAL_INSPECTION     the fixture is expected to be complete NOW. This context comes from OUTSIDE the line: the
                       test protocol or an operator (e.g. a staged CSI Session B fixture), never from the equipment
                       and never inferred from the image. A complete good order has 30 vials (organizer Dataset 1
                       guide), so every empty position, trailing ones included, is MISSED.
An unreadable position is flagged UNSURE on its own and never moves F: spreading UNSURE to every empty before it
flagged 19 extra trailing positions on DEV runs 1-3 (crop-overlap ghosts of the rule baseline, Oct 16).
SequenceTracker adds temporal history: an empty position latches to MISSED once a later position has stayed
occupied for `persist` valid frames; before that it is UNSURE (no one-frame flicker to MISSED).

A placement count K (fill_count) never comes from the line equipment. It is optional: offline metadata (synthetic
scenes, staged ground truth) or the camera's own classifier-free placement-event counter (events.py, video only):
    position <= K : pass -> PASS, defect -> DEFECT, no_vial -> MISSED
    position >  K : no_vial -> WAITING, anything occupied -> UNSURE (nothing should be there yet)
"""
N_POSITIONS = 30
DEFECT_VISUALS = {"missing_cap", "crooked_cap", "damaged_vial"}
VISUAL_STATES = {"pass", "no_vial", "unknown"} | DEFECT_VISUALS
FINAL_STATES = ("PASS", "DEFECT", "MISSED", "WAITING", "UNSURE")
FRONTIER_POLICIES = ("waiting", "unsure")
LIVE_FILL, FINAL_INSPECTION = "live_fill", "final_inspection"
MODES = (LIVE_FILL, FINAL_INSPECTION)


def check_fill_count(fill_count):
    if fill_count is None:
        return None
    if isinstance(fill_count, bool) or not isinstance(fill_count, int) or not 0 <= fill_count <= N_POSITIONS:
        raise ValueError(f"fill count must be an int in 0..{N_POSITIONS} or None, got {fill_count!r}")
    return fill_count


def check_visuals(visuals):
    if len(visuals) != N_POSITIONS:
        raise ValueError(f"need {N_POSITIONS} visual states, got {len(visuals)}")
    bad = [v for v in visuals if v not in VISUAL_STATES]
    if bad:
        raise ValueError(f"unknown visual state {bad[0]!r}")
    return list(visuals)


def occupied(visual):
    return visual not in ("no_vial", "unknown")


def resolve_state(index, fill_count, visual):
    """Count path: index 1..30 (G01 = 1), fill_count = placements so far (K, never from the line equipment).
    Returns (state, reason)."""
    if not 1 <= index <= N_POSITIONS:
        raise ValueError(f"position index must be 1..{N_POSITIONS}, got {index!r}")
    if visual not in VISUAL_STATES:
        raise ValueError(f"unknown visual state {visual!r}")
    K = check_fill_count(fill_count)
    if K is None:
        raise ValueError("resolve_state needs a fill count; use resolve_fixture for occupancy-only reasoning")
    if visual == "unknown":
        return "UNSURE", "vision could not decide"
    if index > K:
        if visual == "no_vial":
            return "WAITING", f"after fill count K={K}"
        return "UNSURE", f"occupied ({visual}) beyond fill count K={K}"
    if visual == "pass":
        return "PASS", ""
    if visual in DEFECT_VISUALS:
        return "DEFECT", visual
    return "MISSED", f"empty inside the filled prefix (K={K})"


def infer_fill_count(visuals):
    """F: the highest position that visibly holds a vial (any visual except no_vial / unknown); 0 if none."""
    return max((i for i, v in enumerate(check_visuals(visuals), 1) if occupied(v)), default=0)


def check_mode(mode, frontier):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if frontier not in FRONTIER_POLICIES:
        raise ValueError(f"frontier policy must be one of {FRONTIER_POLICIES}, got {frontier!r}")


def _empty_after_frontier(i, F, mode, frontier):
    at = f"G{F:02d}" if F else "the start (no vial yet)"
    if mode == FINAL_INSPECTION:
        return "MISSED", f"final inspection (fixture declared complete by the test protocol/operator) but empty after {at}"
    if i == F + 1:
        why = f"fill frontier after {at}: not attempted yet or just skipped; resolves when a later position fills"
        return ("UNSURE" if frontier == "unsure" else "WAITING"), why
    return "WAITING", f"after fill frontier {at}"


def _visual_state(visual):
    if visual == "unknown":
        return "UNSURE", "vision could not decide"
    return ("PASS", "") if visual == "pass" else ("DEFECT", visual)


def resolve_occupancy(visuals, mode=LIVE_FILL, frontier="waiting"):
    """Single frame, no placement count. Returns (F, [(state, reason), ...])."""
    check_mode(mode, frontier)
    visuals = check_visuals(visuals)
    F = infer_fill_count(visuals)
    out = []
    for i, v in enumerate(visuals, 1):
        if v != "no_vial":
            out.append(_visual_state(v))
        elif i < F:
            out.append(("MISSED", f"empty before occupied G{F:02d}"))
        else:
            out.append(_empty_after_frontier(i, F, mode, frontier))
    return F, out


def resolve_fixture(visuals, fill_count=None, mode=LIVE_FILL, frontier="waiting"):
    """All 30 positions. Returns (K_or_F, source, [(state, reason), ...]); source is 'given' (count K: offline
    metadata or the camera's event counter) or 'occupancy' (runtime default: F from what is visible)."""
    if fill_count is not None:
        K = check_fill_count(fill_count)
        return K, "given", [resolve_state(i, K, v) for i, v in enumerate(check_visuals(visuals), 1)]
    F, states = resolve_occupancy(visuals, mode, frontier)
    return F, "occupancy", states


class SequenceTracker:
    """Occupancy reasoning over a sequence of valid (arm-clear) frames of one fixture.

    A position counts as confirmed occupied after `persist` consecutive frames that read it occupied ('unknown'
    leaves its streak unchanged). The confirmed frontier C only grows: the robot fills forward and never removes a
    vial. Empty positions below C are MISSED (latched); an empty position below a later vial that is not yet
    confirmed is UNSURE; a confirmed position that now reads empty is UNSURE (misread or removed vial)."""

    def __init__(self, persist=2, mode=LIVE_FILL, frontier="waiting"):
        if isinstance(persist, bool) or not isinstance(persist, int) or persist < 1:
            raise ValueError(f"persist must be an int >= 1, got {persist!r}")
        check_mode(mode, frontier)
        self.persist, self.mode, self.frontier = persist, mode, frontier
        self.streak = [0] * N_POSITIONS
        self.confirmed = [False] * N_POSITIONS

    @property
    def frontier_confirmed(self):
        return max((i for i, c in enumerate(self.confirmed, 1) if c), default=0)

    def update(self, visuals):
        """visuals: 30 visual states of one valid frame. Returns (C, F, [(state, reason), ...])."""
        visuals = check_visuals(visuals)
        for k, v in enumerate(visuals):
            if v == "unknown":
                continue
            self.streak[k] = self.streak[k] + 1 if occupied(v) else 0
            if self.streak[k] >= self.persist:
                self.confirmed[k] = True
        C, F = self.frontier_confirmed, infer_fill_count(visuals)
        top = max(C, F)
        out = []
        for i, v in enumerate(visuals, 1):
            if v != "no_vial":
                out.append(_visual_state(v))
            elif self.confirmed[i - 1]:
                out.append(("UNSURE", f"G{i:02d} was confirmed occupied, now reads empty"))
            elif i < C:
                out.append(("MISSED", f"empty before confirmed G{C:02d}"))
            elif i < F:
                out.append(("UNSURE", f"empty before G{F:02d}, occupied for fewer than {self.persist} frames; awaiting confirmation"))
            else:
                out.append(_empty_after_frontier(i, top, self.mode, self.frontier))
        return C, F, out
