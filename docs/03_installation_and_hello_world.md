# 03 — Installation & hello world

The supported and recommended runtime is **Docker** (Vulcanexus Humble image). The **hello world** here is minimal — it only confirms the install
works and the node runs. To see real detections, run the **demo** in
[`04_basic_demo_how_to_use.md`](04_basic_demo_how_to_use.md).

## Dependencies

| Category | Hello world (no hardware/bag) | Demo (sample rosbag) | Where |
|---|---|---|---|
| Operating system | Ubuntu 22.04 (via image) | Ubuntu 22.04 | Docker base image |
| ROS 2 / Vulcanexus | Vulcanexus Humble | Vulcanexus Humble | `eprosima/vulcanexus:humble-desktop` |
| Python deps | `ultralytics`, `openvino`, `torch` (CPU), `numpy<2`, `lap` | same | [`Dockerfile`](../Dockerfile) |
| System deps | `python3-opencv`, `ffmpeg` | same | `Dockerfile` |
| ROS deps | `cv_bridge`, `sensor_msgs`, `geometry_msgs`, `rclpy` | same **+ `rosbag2`** | `package.xml` |
| Hardware | **none** | **none** (recorded bag) | — |
| Data | **none** | demo rosbag (external download) | [`docs/04_basic_demo_how_to_use.md`](04_basic_demo_how_to_use.md) |


## Install (Docker)

```bash
# from repository root
docker build -t cv_tool:humble .
```

The image copies `cv_tool_ws/src/`, runs `colcon build --symlink-install`, and sets the default
`CMD` to launch the action server (the hello world below).

## Hello world — start the server (no hardware)

The hello world just proves that everything is installed and the node comes up — **no camera and no
rosbag**.

```bash
docker run --rm -it --net=host cv_tool:humble
```

### Expected output

Within a few seconds the server logs:

```
Logging console output to logs/<DATE>_<TIME>.log
[cv_tool_action_server]: CVToolActionServer starting...
[cv_tool_action_server]: Loading YOLO model from: /.../share/cv_tool/models/11n_int8_openvino_model
[cv_tool_action_server]: CVToolActionServer ready. Waiting for goals...
```

Seeing `CVToolActionServer ready. Waiting for goals...` confirms the dependencies (rclpy, OpenCV,
Ultralytics/OpenVINO/Torch), the bundled model, and the ROS 2 action server all loaded correctly.
Stop with `Ctrl-C`.

**Optional confirmation** that the action interface is alive — in a second shell into the container
(`docker exec -it <container_id> bash`, then `source /cv_tool_ws/install/setup.bash`):

```bash
ros2 action info /detect_tool -t          # lists cv_tool_action_server as the action server
# or send a goal (with no camera/bag this aborts — expected):
ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect "{tool_name: screwdriver}"
#   -> server warns "No camera feed available" and the goal is aborted. This is expected without
#      input data and still proves the server received and handled the goal.
```

## Native (no Docker)

```bash
cd cv_tool_ws
source /opt/ros/humble/setup.bash       # or Vulcanexus setup
pip3 install ultralytics "numpy<2.0.0" "lap>=0.5.12" "openvino>=2024.0.0" \
    torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu
colcon build --symlink-install
source install/setup.bash
ros2 launch cv_tool cv_tool.launch.py    # hello world: starts the server (no camera/bag)
```

## Next: the demo

To exercise real detection end-to-end (download + replay the sample rosbag, send goals, read the
3D results), continue to [`04_basic_demo_how_to_use.md`](04_basic_demo_how_to_use.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Goal aborts: *"No camera feed available"* | Expected in the hello world (no input). In the **demo**, the bag isn't playing or topics don't match — check `ros2 topic hz /camera/camera/color/image_raw`, `bag_path`, and that replay started. |
| `ros2 bag play` errors about missing metadata (demo) | Run `ros2 bag reindex /cv_tool_ws/rosbags -s sqlite3`. |
| Warning: *"No depth feed available"* (demo) | Depth topic missing/misnamed; 3D points/size disambiguation will be unreliable. Check `depth_topic`. |
| Goal never succeeds (demo) | The target tool may not be present/centered in the loop window; try another `tool_name`, lower `conf_thres`, or raise `margin_x/y`. |
| Nodes don't see each other | DDS discovery — use `--net=host`; align `ROS_DOMAIN_ID` across shells. |
| Set `verbose: true` | Saves annotated RGB+depth frames to `output_images/` for inspection. |
