# Screenshots

Annotated RGB+depth composite frames produced by running the demo with `verbose: true`
(saved to the mounted output folder). The three frames below show the typical detection
progression for an `allen_large` goal:

| File | Status overlay | What it shows |
|---|---|---|
| `20260617_095712_482013_allen_large_0040.jpg` | Scanning — Not Found | Other tools visible; allen_large not yet in view |
| `20260617_095755_948306_allen_large_0659.jpg` | Found but not centered #24 | allen detected (19×9 cm, d=47 cm) but outside margin |
| `20260617_095759_821694_allen_large_0717.jpg` | Found and Centered! | Stable, centered detection — goal succeeds |

Each frame is a side-by-side RGB (annotated bounding boxes, class + confidence, `WxH cm, d=cm`
size label, centre crosshair) and depth map (normalised to 8-bit).

Referenced from [`docs/04_basic_demo_how_to_use.md`](../../docs/04_basic_demo_how_to_use.md)
and [`README.md`](../../README.md).
