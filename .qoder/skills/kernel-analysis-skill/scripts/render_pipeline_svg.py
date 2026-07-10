#!/usr/bin/env python3
"""Render a FlashAttention-v3-style logical pipeline timeline as standalone SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


COLORS = {
    "transfer": ("#c9e6f5", "data movement"),
    "compute": ("#f4c7c3", "tensor/core compute"),
    "epilogue": ("#d8e8c6", "vector / reduction / epilogue"),
    "other": ("#f7deb5", "other work"),
}


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _load_spec(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise ValueError("top-level JSON value must be an object")
    lanes = spec.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        raise ValueError("lanes must be a non-empty list")

    max_time = 0.0
    for lane_index, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            raise ValueError(f"lanes[{lane_index}] must be an object")
        _text(lane.get("label"), f"lanes[{lane_index}].label")
        events = lane.get("events")
        if not isinstance(events, list):
            raise ValueError(f"lanes[{lane_index}].events must be a list")
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(f"lanes[{lane_index}].events[{event_index}] must be an object")
            start = _number(event.get("start"), f"event {lane_index}:{event_index}.start")
            end = _number(event.get("end"), f"event {lane_index}:{event_index}.end")
            if end <= start:
                raise ValueError(f"event {lane_index}:{event_index}.end must be greater than start")
            _text(event.get("label"), f"event {lane_index}:{event_index}.label")
            if event.get("kind") not in COLORS:
                raise ValueError(f"event {lane_index}:{event_index}.kind must be one of {', '.join(COLORS)}")
            max_time = max(max_time, end)

    sync = spec.get("sync", [])
    if not isinstance(sync, list):
        raise ValueError("sync must be a list")
    for index, marker in enumerate(sync):
        if not isinstance(marker, dict):
            raise ValueError(f"sync[{index}] must be an object")
        max_time = max(max_time, _number(marker.get("time"), f"sync[{index}].time"))
        _text(marker.get("label"), f"sync[{index}].label")
    if max_time == 0:
        raise ValueError("the timeline needs at least one non-zero event or sync time")
    return spec


def render(spec: dict[str, Any]) -> str:
    lanes: list[dict[str, Any]] = spec["lanes"]
    max_time = max(
        [event["end"] for lane in lanes for event in lane["events"]]
        + [marker["time"] for marker in spec.get("sync", [])]
    )
    left, right, top, lane_height = 205, 55, 72, 74
    width = 1220
    height = top + len(lanes) * lane_height + 98
    scale = (width - left - right) / max_time
    axis_y = top + len(lanes) * lane_height + 15
    esc = lambda value: html.escape(str(value), quote=True)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<defs><marker id=\"arrow\" markerWidth=\"9\" markerHeight=\"7\" refX=\"8\" refY=\"3.5\" orient=\"auto\"><path d=\"M0,0 L9,3.5 L0,7 Z\" fill=\"#111\"/></marker></defs>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: Arial, &quot;Microsoft YaHei&quot;, sans-serif; fill: #151515; } .label { font-size: 16px; font-weight: 600; } .event { font-size: 14px; } .small { font-size: 12px; } .sync { stroke: #202020; stroke-width: 1.5; stroke-dasharray: 7 6; }</style>',
        f'<text x="{left}" y="29" class="label">{esc(spec.get("title", "Kernel pipeline"))}</text>',
    ]

    legend_x = left + 270
    for index, (kind, (color, label)) in enumerate(COLORS.items()):
        x = legend_x + index * 170
        parts.append(f'<rect x="{x}" y="15" width="14" height="14" rx="2" fill="{color}" stroke="#222"/>')
        parts.append(f'<text x="{x + 20}" y="27" class="small">{esc(label)}</text>')

    for marker in spec.get("sync", []):
        x = left + marker["time"] * scale
        parts.append(f'<line x1="{x:.1f}" y1="47" x2="{x:.1f}" y2="{axis_y - 10}" class="sync"/>')
        parts.append(f'<text x="{x + 5:.1f}" y="60" class="small">{esc(marker["label"])}</text>')

    for lane_index, lane in enumerate(lanes):
        y = top + lane_index * lane_height
        parts.append(f'<text x="{left - 15}" y="{y + 25}" text-anchor="end" class="label">{esc(lane["label"])}</text>')
        parts.append(f'<line x1="{left}" y1="{y + 32}" x2="{width - right}" y2="{y + 32}" stroke="#dedede"/>')
        for event in lane["events"]:
            x = left + event["start"] * scale
            event_width = (event["end"] - event["start"]) * scale
            color = COLORS[event["kind"]][0]
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{event_width:.1f}" height="48" rx="2" fill="{color}" stroke="#222" stroke-width="1.2"/>')
            parts.append(f'<text x="{x + event_width / 2:.1f}" y="{y + 29}" text-anchor="middle" class="event">{esc(event["label"])}</text>')

    parts.extend(
        [
            f'<line x1="{left}" y1="{axis_y}" x2="{width - right + 12}" y2="{axis_y}" stroke="#111" stroke-width="1.5" marker-end="url(#arrow)"/>',
            f'<text x="{(left + width - right) / 2:.1f}" y="{axis_y + 31}" text-anchor="middle" class="label">{esc(spec.get("time_label", "logical time"))}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="pipeline JSON specification")
    parser.add_argument("--out", required=True, type=Path, help="output SVG path")
    args = parser.parse_args()
    spec = _load_spec(args.spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(spec), encoding="utf-8")


if __name__ == "__main__":
    main()
