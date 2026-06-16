# 03 — Installation & hello world

The supported and recommended runtime is **Docker** (Vulcanexus Humble image). A native ROS 2
Humble path is also documented.

## Dependencies

| Category | Hello world (rosbag replay) | Full demo (live camera) | Where |
|---|---|---|---|
| Operating system | Ubuntu 22.04 (via image) | Ubuntu 22.04 | Docker base image |
| ROS 2 / Vulcanexus | Vulcanexus Humble | Vulcanexus Humble | `eprosima/vulcanexus:humble-desktop` |
| Python deps | `ultralytics`, `openvino`, `torch` (CPU), `numpy<2`, `lap` | same | [`Dockerfile`](../Dockerfile) |
| System deps | `python3-opencv`, `ffmpeg` | same | `Dockerfile` |
| ROS deps | `cv_bridge`, `sensor_msgs`, `geometry_msgs`, `rclpy`, `rosbag2` | same (minus rosbag2) | `package.xml` |
| Hardware | **none** (recorded bag) | Intel RealSense (or compatible) RGB-D camera | — |
| Data | demo rosbag (external download) | live camera | [`examples/bags/README.md`](../cv_tool_ws/src/cv_tool/examples/bags/README.md) |

> Why Docker: `ultralytics` / `openvino` / `torch` are pip-only (no rosdep keys) and are pinned in
> the Dockerfile, so the container gives a reproducible environment without manual dependency setup.

## Install (Docker)

```bash
# from repository root
docker build -t cv_tool:humble .
```

The image copies `cv_tool_ws/src/`, runs `colcon build --symlink-install`, and sets the default
`CMD` to launch the live-camera node.

## Hello world — replay the demo rosbag (no hardware)

1. **Download the demo rosbag** (`boxes_0.db3`) — see
   [`examples/bags/README.md`](../cv_tool_ws/src/cv_tool/examples/bags/README.md) for the link and
   details (RealSense RGB-D, ~71 s @ 15 Hz, topics already matching the config).

2. **Run the container, mounting the bag folder:**

   ```bash
   docker run --rm -it --net=host \
       -v /absolute/path/to/rosbags:/cv_tool_ws/bags \
       cv_tool:humble bash
   ```

3. **(Once, if needed) regenerate the bag index** if the download has no `metadata.yaml`:

   ```bash
   ros2 bag reindex /cv_tool_ws/bags -s sqlite3
   ```

4. **Replay the bag and start the server:**

   ```bash
   source /cv_tool_ws/install/setup.bash
   ros2 launch cv_tool cv_tool_replay.launch.py bag_path:=/cv_tool_ws/bags/boxes_0.db3
   ```

5. **Send a goal** from a second shell into the same container:

   ```bash
   docker exec -it <container_id> bash
   source /cv_tool_ws/install/setup.bash
   ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect \
       "{tool_name: screwdriver}" --feedback
   ```

### Expected output

Server log:

```
[cv_tool_action_server]: CVToolActionServer starting...
[cv_tool_action_server]: Loading YOLO model from: /.../share/cv_tool/models/11n_int8_openvino_model
[cv_tool_action_server]: CVToolActionServer ready. Waiting for goals...
[cv_tool_action_server]: Received goal to detect: screwdriver
[cv_tool_action_server]: screwdriver found and centered!
```

Action client:

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

A successful goal is the evidence that installation works. (Try other tools from
`tool_class_names`; the bag is named *boxes* and contains the KIRO tool set in trays.)

## Native (no Docker)

```bash
cd cv_tool_ws
source /opt/ros/humble/setup.bash       # or Vulcanexus setup
pip3 install ultralytics "numpy<2.0.0" "lap>=0.5.12" "openvino>=2024.0.0" \
    torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu
colcon build --symlink-install
source install/setup.bash
ros2 launch cv_tool cv_tool_replay.launch.py bag_path:=/path/to/boxes_0.db3
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Goal aborts: *"No camera feed available"* | The bag isn't playing or topics don't match. Check `ros2 topic hz /camera/camera/color/image_raw`; verify `bag_path` and that replay started. |
| `ros2 bag play` errors about missing metadata | Run `ros2 bag reindex /cv_tool_ws/bags -s sqlite3`. |
| Warning: *"No depth feed available"* | Depth topic missing/misnamed; 3D points/size disambiguation will be unreliable. Check `depth_topic`. |
| Goal never succeeds | The target tool may not be present/centered in the loop window; try another `tool_name`, lower `conf_thres`, or raise `margin_x/y`. |
| Nodes don't see each other | DDS discovery — use `--net=host`; align `ROS_DOMAIN_ID` across shells. |
| Set `verbose: true` | Saves annotated RGB+depth frames to `output_images_<model>/` for inspection. |
