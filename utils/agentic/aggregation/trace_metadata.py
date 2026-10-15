"""Trace metadata lookup for agentic aggregate generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_TRACE_METADATA_CACHE: dict[str, list[dict[str, Any]]] | None = None


def conversation_id_to_trace_id(conv_id: str | None) -> str | None:
    """Strip aiperf's ``::sa:<agent>`` suffix to recover the parent trace id."""
    if not conv_id:
        return None
    return conv_id.split("::", 1)[0]


def _hf_traces_dir() -> Path | None:
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        cache_root = Path(hub_cache)
    else:
        home = os.environ.get("HF_HOME")
        cache_root = Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_root.is_dir():
        return None

    snapshots: list[Path] = []
    for dataset_dir in cache_root.glob("datasets--semianalysisai--cc-traces-weka*"):
        snap_root = dataset_dir / "snapshots"
        if not snap_root.is_dir():
            continue
        snapshots.extend(p for p in snap_root.iterdir() if p.is_dir())
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for candidate in snapshots:
        if any(candidate.glob("*.jsonl")) or any(candidate.glob("*.json")):
            return candidate
    return None


def _iter_trace_blobs(traces_dir: Path):
    for path in sorted(traces_dir.glob("*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    for path in sorted(traces_dir.glob("*.json")):
        try:
            with open(path) as f:
                yield json.load(f)
        except (json.JSONDecodeError, OSError):
            continue


def _request_metadata(req: dict[str, Any]) -> dict[str, Any]:
    output_length = req.get("output_length")
    if output_length is None:
        output_length = req.get("out")
    return {
        "type": req.get("type"),
        "output_length": int(output_length) if output_length is not None else None,
        "requests": [_request_metadata(child) for child in req.get("requests", [])],
    }


def load_trace_metadata() -> dict[str, list[dict[str, Any]]]:
    """Load output lengths without changing original outer or inner indices."""
    global _TRACE_METADATA_CACHE
    if _TRACE_METADATA_CACHE is not None:
        return _TRACE_METADATA_CACHE

    out: dict[str, list[dict[str, Any]]] = {}
    traces_dir = _hf_traces_dir()
    if traces_dir is None:
        _TRACE_METADATA_CACHE = out
        return out

    for blob in _iter_trace_blobs(traces_dir):
        trace_id = blob.get("id")
        if not trace_id:
            continue
        per_turn = [_request_metadata(req) for req in blob.get("requests", [])]
        if per_turn:
            out[str(trace_id)] = per_turn

    _TRACE_METADATA_CACHE = out
    return out


def expected_output_lengths(records: list[dict[str, Any]]) -> list[int]:
    metadata = load_trace_metadata()
    if not metadata:
        return []

    expected: list[int] = []
    for record in records:
        record_metadata = record.get("metadata", {})
        source_trace_id = record_metadata.get("source_trace_id")
        outer_idx = record_metadata.get("source_outer_idx")
        if source_trace_id is not None and outer_idx is not None:
            # Replay turn indices restart after slicing and within child sessions.
            turns = metadata.get(str(source_trace_id), [])
            if not 0 <= outer_idx < len(turns):
                continue
            req = turns[outer_idx]
            inner_idx = record_metadata.get("source_inner_idx")
            if inner_idx is not None:
                children = req["requests"]
                if not 0 <= inner_idx < len(children):
                    continue
                req = children[inner_idx]
            if req["output_length"] is not None:
                # Weka raises zero-length outputs to one sendable token.
                expected.append(max(1, req["output_length"]))
            continue

        # Retain the lookup for older exports without source indices.
        trace_id = conversation_id_to_trace_id(record_metadata.get("conversation_id"))
        turn_index = record_metadata.get("turn_index")
        if trace_id is None or turn_index is None:
            continue
        turns = [req for req in metadata.get(trace_id, []) if req["type"] in ("n", "s")]
        if not 0 <= turn_index < len(turns):
            continue
        expected.append(int(turns[int(turn_index)]["output_length"] or 0))
    return expected
