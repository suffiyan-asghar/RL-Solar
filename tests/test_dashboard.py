"""Drive the Streamlit dashboard headlessly (streamlit.testing)."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "dashboard.py")


def test_dashboard_runs_and_plays():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    assert len(at.session_state["results"]) == 2
    n = len(at.session_state["results"][0][2])
    assert n == 168
    # jump to the end, then restart
    [b for b in at.button if "End" in b.label][0].click().run()
    assert not at.exception
    assert at.session_state["hour"] == n - 1
    [b for b in at.button if "Restart" in b.label][0].click().run()
    assert at.session_state["hour"] == 0
    # +24 h
    [b for b in at.button if "+24" in b.label][0].click().run()
    assert at.session_state["hour"] == 24


def test_dashboard_sliders_change_results():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    cost0 = at.session_state["results"][0][1]["cost"]
    exp = [s for s in at.slider if s.label.startswith("Export rate")][0]
    exp.set_value(30.0)
    [b for b in at.sidebar.button if b.label == "Run"][0].click().run()
    assert not at.exception, at.exception
    cost1 = at.session_state["results"][0][1]["cost"]
    exported = at.session_state["results"][0][1]["export_kwh"]
    if exported > 0:
        assert cost1 < cost0


def test_dashboard_play_animates_to_end():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    [s for s in at.select_slider if s.label.startswith("Speed")][0].set_value(6).run()
    [b for b in at.button if "Play" in b.label][0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["hour"] == 167 and not at.session_state["playing"]
