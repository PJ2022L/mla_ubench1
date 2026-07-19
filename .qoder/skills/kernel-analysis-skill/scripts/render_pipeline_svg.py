#!/usr/bin/env python3
"""Render a FlashAttention-v3-style logical pipeline timeline as standalone SVG."""

from __future__ import annotations

import argparse
import html
import json
import unicodedata
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


def _display_units(value: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in value)


def _take_units(value: str, limit: int) -> tuple[str, str]:
    used = 0
    for index, char in enumerate(value):
        width = 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        if used + width > limit:
            return value[:index], value[index:]
        used += width
    return value, ""


def _event_lines(label: str, pixel_width: float) -> tuple[list[str], int]:
    font_size = 13 if pixel_width >= 120 else 12 if pixel_width >= 80 else 10
    max_units = max(3, int((pixel_width - 12) / (font_size * 0.58)))
    words = label.split()
    lines: list[str] = []
    current = ""

    while words:
        word = words.pop(0)
        candidate = word if not current else f"{current} {word}"
        if _display_units(candidate) <= max_units:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
            words.insert(0, word)
        else:
            head, tail = _take_units(word, max_units)
            lines.append(head)
            if tail:
                words.insert(0, tail)
        if len(lines) == 2:
            break

    if len(lines) < 2 and current:
        lines.append(current)
        current = ""
    if words or current:
        last, _ = _take_units(lines[-1], max(1, max_units - 1))
        lines[-1] = f"{last}..."
    return lines or [label], font_size


def _sync_layout(
    markers: list[dict[str, Any]], left: int, scale: float, chart_right: int
) -> tuple[list[dict[str, Any]], int]:
    rows_end: list[float] = []
    laid_out: list[dict[str, Any]] = []
    for marker in sorted(markers, key=lambda item: item["time"]):
        x = left + marker["time"] * scale
        label_width = min(280.0, _display_units(marker["label"]) * 7.0)
        if x + 7 + label_width <= chart_right:
            text_x = x + 7
            anchor = "start"
            interval = (text_x, text_x + label_width)
        else:
            text_x = x - 7
            anchor = "end"
            interval = (text_x - label_width, text_x)

        row = 0
        while row < len(rows_end) and interval[0] <= rows_end[row] + 14:
            row += 1
        if row == len(rows_end):
            rows_end.append(interval[1])
        else:
            rows_end[row] = interval[1]
        laid_out.append(
            {
                **marker,
                "x": x,
                "text_x": text_x,
                "anchor": anchor,
                "row": row,
                "label_left": interval[0],
                "label_width": interval[1] - interval[0],
            }
        )
    return laid_out, max(1, len(rows_end))


def render(spec: dict[str, Any]) -> str:
    lanes: list[dict[str, Any]] = spec["lanes"]
    max_time = max(
        [event["end"] for lane in lanes for event in lane["events"]]
        + [marker["time"] for marker in spec.get("sync", [])]
    )
    left, right, lane_height = 225, 60, 82
    width = 1440
    scale = (width - left - right) / max_time
    sync_layout, sync_rows = _sync_layout(spec.get("sync", []), left, scale, width - right)
    sync_label_y = 82
    top = sync_label_y + sync_rows * 18 + 14
    height = top + len(lanes) * lane_height + 98
    axis_y = top + len(lanes) * lane_height + 15
    esc = lambda value: html.escape(str(value), quote=True)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<defs><marker id=\"arrow\" markerWidth=\"9\" markerHeight=\"7\" refX=\"8\" refY=\"3.5\" orient=\"auto\"><path d=\"M0,0 L9,3.5 L0,7 Z\" fill=\"#111\"/></marker></defs>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: Arial, &quot;Microsoft YaHei&quot;, sans-serif; fill: #151515; letter-spacing: 0; } .title { font-size: 17px; font-weight: 600; } .label { font-size: 15px; font-weight: 600; } .small { font-size: 12px; } .sync-label { font-size: 12px; } .sync { stroke: #202020; stroke-width: 1.4; stroke-dasharray: 7 6; }</style>',
        f'<text x="{left}" y="27" class="title">{esc(spec.get("title", "Kernel pipeline"))}</text>',
    ]

    legend_x = left
    for index, (kind, (color, label)) in enumerate(COLORS.items()):
        x = legend_x + index * 210
        parts.append(f'<rect x="{x}" y="42" width="14" height="14" rx="2" fill="{color}" stroke="#222"/>')
        parts.append(f'<text x="{x + 20}" y="54" class="small">{esc(label)}</text>')

    for marker in sync_layout:
        y = sync_label_y + marker["row"] * 18
        parts.append(f'<line x1="{marker["x"]:.1f}" y1="66" x2="{marker["x"]:.1f}" y2="{axis_y - 10}" class="sync"/>')
        parts.append(
            f'<rect x="{marker["label_left"] - 3:.1f}" y="{y - 13}" width="{marker["label_width"] + 6:.1f}" height="17" rx="2" fill="white" fill-opacity="0.94"/>'
        )
        parts.append(
            f'<text x="{marker["text_x"]:.1f}" y="{y}" text-anchor="{marker["anchor"]}" class="sync-label">{esc(marker["label"])}</text>'
        )

    event_index = 0
    for lane_index, lane in enumerate(lanes):
        y = top + lane_index * lane_height
        parts.append(f'<text x="{left - 15}" y="{y + 31}" text-anchor="end" class="label">{esc(lane["label"])}</text>')
        parts.append(f'<line x1="{left}" y1="{y + 36}" x2="{width - right}" y2="{y + 36}" stroke="#dedede"/>')
        for event in lane["events"]:
            x = left + event["start"] * scale
            event_width = (event["end"] - event["start"]) * scale
            color = COLORS[event["kind"]][0]
            clip_id = f"event-clip-{event_index}"
            lines, font_size = _event_lines(event["label"], event_width)
            line_height = font_size + 2
            first_y = y + 29 - (len(lines) - 1) * line_height / 2
            parts.append(f'<clipPath id="{clip_id}"><rect x="{x + 3:.1f}" y="{y + 3}" width="{max(0.0, event_width - 6):.1f}" height="48" rx="1"/></clipPath>')
            parts.append(f'<g><title>{esc(event["label"])}</title><rect x="{x:.1f}" y="{y}" width="{event_width:.1f}" height="54" rx="3" fill="{color}" stroke="#222" stroke-width="1.2"/>')
            parts.append(f'<text x="{x + event_width / 2:.1f}" y="{first_y:.1f}" text-anchor="middle" font-size="{font_size}px" clip-path="url(#{clip_id})">')
            for line_index, line in enumerate(lines):
                dy = 0 if line_index == 0 else line_height
                parts.append(f'<tspan x="{x + event_width / 2:.1f}" dy="{dy}">{esc(line)}</tspan>')
            parts.append('</text></g>')
            event_index += 1

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
