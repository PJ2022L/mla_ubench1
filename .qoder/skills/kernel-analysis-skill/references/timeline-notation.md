# Pipeline timeline notation

Use a separate timeline for each level of overlap. The position and width of a box communicate logical order and overlap; they are not cycle-accurate unless sourced from a trace.

## Required appearance

- Draw time left-to-right, with a labelled arrow on the horizontal axis.
- Use one lane per warp group/role for the outer pipeline; use one lane per operation stream for the inner pipeline.
- Use **blue** (`transfer`) for global/shared-memory movement, **pink** (`compute`) for tensor/core work, and **green** (`epilogue`) for vector, reduction, softmax, normalization, or writeback work. Use amber (`other`) only for distinct control work.
- Draw a **vertical dashed line** at each synchronization/dependence boundary. Supply a short label such as `TMA ready (mbarrier)` or `accumulator ready (wait_group 0)`.
- Put the loop iteration and buffer/stage index in each event label: `TMA K/V[2]`, `WGMMA[1]`, `softmax[1]`.
- Repeat the first two iterations and the steady state only; use an ellipsis for a long loop. Do not draw a fabricated final-drain phase.

## SVG generator input

Run the generator from the skill directory or provide absolute paths:

```powershell
python scripts/render_pipeline_svg.py --spec outer.json --out analysis/warpgroup-pipeline.svg
python scripts/render_pipeline_svg.py --spec inner.json --out analysis/intra-warpgroup-pipeline.svg
```

The input is JSON:

```json
{
  "title": "Warp-group pipeline",
  "time_label": "logical time",
  "lanes": [
    {
      "label": "Warp group 0 (consumer)",
      "events": [
        {"start": 1, "end": 3, "label": "WGMMA[0]", "kind": "compute"},
        {"start": 3, "end": 4, "label": "softmax[0]", "kind": "epilogue"}
      ]
    },
    {
      "label": "Warp group 1 (producer)",
      "events": [
        {"start": 0, "end": 1, "label": "TMA K/V[0]", "kind": "transfer"}
      ]
    }
  ],
  "sync": [
    {"time": 1, "label": "K/V[0] ready (mbarrier)"},
    {"time": 3, "label": "accumulator ready (wait_group)"}
  ]
}
```

`start`, `end`, and `time` must be non-negative logical-time values; every event needs `end > start`. Valid `kind` values are `transfer`, `compute`, `epilogue`, and `other`.

## Choosing diagrams

- Use the outer diagram whenever two or more execution roles or warp groups overlap, including a producer/consumer TMA pipeline.
- Use the inner diagram only for real independent streams within one warp group, for example WGMMA issue versus vector softmax/reduction or a second MMA stream.
- When the source places `wait_group`, a barrier, or a data dependency between the candidate stages, emit only the outer diagram and document the serial chain. A FlashAttention-style reference image is not evidence that another kernel has the same overlap.
