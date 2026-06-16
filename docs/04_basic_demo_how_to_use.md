# 04 — Basic demo & how to use

The hello world ([`03`](03_installation_and_hello_world.md)) proves the install works. This page
shows how to *use* the module: configure it, request different tools, and read its output. All
commands run inside the Docker container with the demo rosbag replaying
(`ros2 launch cv_tool cv_tool_replay.launch.py bag_path:=/cv_tool_ws/bags/boxes_0.db3`).

## Scenario A — request different tools

Send goals for any name in `tool_class_names`:

```bash
ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect "{tool_name: multimeter}" --feedback
ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect "{tool_name: combination_wrench}" --feedback
ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect "{tool_name: tape_measure}" --feedback
```

Each goal streams `Scanning frame N ...` feedback until the requested tool is detected and centered,
then returns `success: true` with the 3D keypoints and confidence. Requesting a tool that is not in
the current view simply keeps scanning (cancel with `Ctrl-C`).

## Scenario B — size disambiguation (small vs large)

The model emits a generic `allen` class; the module resolves the size from the depth-derived metric
width/height. Request the small and large variants and compare results:

```bash
ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect "{tool_name: allen_small}" --feedback
ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect "{tool_name: allen_large}" --feedback
```

Only the variant whose measured size matches (threshold ≈ 0.17 m width/height) is accepted. With
`verbose: true`, the saved annotated frame shows the measured `WxH cm, d=cm` label per detection.

## Scenario C — centering-stability behaviour

A detection is accepted only when the tool stays centered within `margin_x`/`margin_y` pixels for
`buffer_size` consecutive frames (defaults 70/70 px, 8 frames). To see the effect, edit
[`config/config.yaml`](../cv_tool_ws/src/cv_tool/config/config.yaml):

- Lower `buffer_size` → accepts faster, less stable.
- Tighten `margin_x/y` → requires the tool nearer the image centre (as during the arm sweep in the
  demonstrator).

Rebuild/re-source after editing (or mount an external config and pass its path with `--config`).

## Reading the output

| Field | Meaning |
|---|---|
| `success` | `true` only on a stable, centered detection above `conf_thres`. |
| `center` / `top_left` / `bottom_right` | 3D points (metres) in the camera color optical frame, for the grasp step. |
| `confidence` | YOLO confidence of the accepted detection. |
| feedback `current_status` | Live scan progress. |

With `verbose: true`, annotated RGB+depth composites are written to `output_images_<model>/`
(bounding boxes, class+confidence, measured size, centre marker, goal/status overlay). Example
annotated frames are in [`../media/screenshots/`](../media/screenshots/).

## Tips

- The CV result is **frame-relative**; the demonstrator pairs it with the arm sweep (it stops the
  motion when the tool is centered). Standalone, expect the goal to succeed whenever the bag loop
  brings the requested tool through the centred region.
- For a different camera, update `camera_intrinsics` and the topic names in the config.
