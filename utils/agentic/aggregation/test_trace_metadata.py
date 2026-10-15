from __future__ import annotations

import json

import pytest

from utils.agentic.aggregation import trace_metadata
from utils.agentic.aggregation.request_metrics import compute_workload_stats


@pytest.fixture
def trace_cache(tmp_path, monkeypatch):
    snapshot = tmp_path / "datasets--semianalysisai--cc-traces-weka-test/snapshots/test"
    snapshot.mkdir(parents=True)
    trace = {
        "id": "original",
        "requests": [
            {"type": "n", "out": 10},
            {"type": "p", "requests": [{"type": "n", "out": 30}, {"type": "n", "out": 0}]},
            {"type": "s", "output_length": 50},
        ],
    }
    (snapshot / "traces.jsonl").write_text(json.dumps(trace) + "\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(trace_metadata, "_TRACE_METADATA_CACHE", None)


def test_sliced_and_child_requests_use_source_indices(trace_cache):
    records = [
        {
            "metadata": {
                "conversation_id": "replayed",
                "turn_index": 0,
                "source_trace_id": "original",
                "source_outer_idx": outer,
                "source_inner_idx": inner,
            }
        }
        for outer, inner in [(2, None), (1, 0), (1, 1), (0, None)]
    ]
    assert trace_metadata.expected_output_lengths(records) == [50, 30, 1, 10]
    _, tokens = compute_workload_stats(records)
    assert tokens["output_expected"]["mean"] == 22.75
    assert tokens["output_expected"]["p50"] == 20


@pytest.mark.parametrize("outer,inner", [(-1, None), (3, None), (1, -1), (1, 2), (1, None)])
def test_invalid_source_does_not_fall_back_to_replay_index(trace_cache, outer, inner):
    record = {
        "metadata": {
            "conversation_id": "original",
            "turn_index": 0,
            "source_trace_id": "original",
            "source_outer_idx": outer,
            "source_inner_idx": inner,
        }
    }
    assert trace_metadata.expected_output_lengths([record]) == []


def test_legacy_exports_keep_filtered_turn_lookup(trace_cache):
    records = [
        {"metadata": {"conversation_id": "original::sa:child", "turn_index": 1}},
        {"metadata": {"conversation_id": "original", "turn_index": 0}},
    ]
    assert trace_metadata.expected_output_lengths(records) == [50, 10]


def test_missing_trace_cache_returns_no_expected_lengths(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(trace_metadata, "_TRACE_METADATA_CACHE", None)
    record = {"metadata": {"source_trace_id": "missing", "source_outer_idx": 0}}
    assert trace_metadata.expected_output_lengths([record]) == []
