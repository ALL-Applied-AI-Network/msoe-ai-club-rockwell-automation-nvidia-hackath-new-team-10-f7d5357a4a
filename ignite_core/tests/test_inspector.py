import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inspector as I


def test_mock_inspector_fails_closed_without_visual_evidence():
    observation = I.MockInspector().inspect(None)
    assert observation == I.Observation(
        visual_state="unknown",
        score=None,
        features=None,
        reason="mock_no_evidence",
    )


def test_mock_inspector_satisfies_protocol_shape():
    assert callable(getattr(I.MockInspector(), "inspect", None))
