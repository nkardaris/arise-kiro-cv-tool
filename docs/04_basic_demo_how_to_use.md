# 04 — Basic demo & how to use

The hello world ([`03`](03_installation_and_hello_world.md)) proves the install works. The **demo**
shows the module producing real detections — without any hardware — by replaying a recorded
RealSense RGB-D rosbag. This page covers the demo setup and then how to *use* the module: request
different tools, exercise size disambiguation and centering, and read the output.

## Set up the demo

1. **Download the sample rosbag** (`boxes_0.db3`) — link and details (RealSense RGB-D, ~71 s @ 15 Hz,
   topics already matching the config) in
   [`examples/bags/README.md`](../cv_tool_ws/src/cv_tool/examples/bags/README.md).

2. **Run the container with the bag folder mounted:**

   ```bash
   docker run --rm -it --net=host \
       -v /absolute/path/to/rosbags:/cv_tool_ws/bags \
       cv_tool:humble bash
   ```

3. **(Once, if the download has no `metadata.yaml`)** regenerate the index:

   ```bash
   ros2 bag reindex /cv_tool_ws/bags -s sqlite3
   ```

4. **Replay the bag and start the server:**

   ```bash
   source /cv_tool_ws/install/setup.bash
   ros2 launch cv_tool cv_tool_demo.launch.py bag_path:=/cv_tool_ws/bags/boxes_0.db3
   ```

5. **Send a goal** from a second shell into the same container
   (`docker exec -it <container_id> bash` → `source /cv_tool_ws/install/setup.bash`):

   ```bash
   ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect \
       "{tool_name: screwdriver}" --feedback
   ```

   Expected server log:

   ```
   [cv_tool_action_server]: Received goal to detect: screwdriver
   [cv_tool_action_server]: screwdriver found and centered!
   ```

   Expected action client result:

   ```
   Feedback: current_status: 'Scanning frame 12 for screwdriver...'
   ...
   Result:
     success: true
     center: {x: ..., y: ..., z: ...}
     top_left: {x: ..., y: ..., z: ...}
     bottom_right: {x: ..., y: ..., z: ...}
     confidence: 0.9...
   ```

The scenarios below all run with this replay active.

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
