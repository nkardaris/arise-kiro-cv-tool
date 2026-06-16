# Screenshots

Place annotated detection screenshots here for the README, docs and D4 report.

Recommended captures (produced by running with `verbose: true`, saved under
`output_images_<model>/`):

- A **"Found and Centered!"** frame for a clearly detected tool (e.g. `screwdriver`,
  `combination_wrench`) showing the bounding box, class + confidence, measured `WxH cm, d=cm`
  size label, the centre marker, and the goal/status overlay.
- A **size-disambiguation** frame showing `allen_small` vs `allen_large` resolved by measured size.
- An RGB + depth composite (the `verbose` output concatenates the annotated RGB with the depth map).

Suggested filenames: `found_and_centered.png`, `allen_size_disambiguation.png`,
`rgb_depth_composite.png`. Reference them from [`docs/04_basic_demo_how_to_use.md`](../../docs/04_basic_demo_how_to_use.md).
