# cv_tool (ROS 2 + Docker)

Action server for image-based tool detection used in the ARISE-KIRO project.
The node subscribes to RGB and depth camera topics, runs a YOLO OpenVINO model, and serves detection results through a ROS 2 action.

## Quick Start

Use these commands from repository root to build and run immediately.

```bash
# 1) Build Docker image
docker build -t cv_tool:humble .

# 2) Run container with ROS 2 discovery over host network
docker run --rm -it --net=host cv_tool:humble
```

The container starts:

```bash
ros2 launch cv_tool cv_tool.launch.py
```

In a second terminal (inside the same container or another ROS 2 shell with the workspace sourced), send a test action goal:

```bash
source /cv_tool_ws/install/setup.bash
ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect "{tool_name: screwdriver}" --feedback
```

If your setup uses a non-default ROS domain, export it before launching in every ROS 2 terminal:

```bash
export ROS_DOMAIN_ID=<your_domain_id>
```

## What this package does

- Runs a ROS 2 action server named `cv_tool_action_server`.
- Exposes action endpoint: `/detect_tool` with action type `cv_tool_interfaces/action/Detect`.
- Subscribes to:
  - RGB image topic
  - Depth image topic
- Detects a requested tool and returns:
  - 3D center point
  - 3D top-left point
  - 3D bottom-right point
  - confidence score

## Repository layout

- `Dockerfile`: containerized build/runtime (ROS 2 Humble on Vulcanexus image).
- `cv_tool_ws/src/cv_tool`: Python node and launch file.
- `cv_tool_ws/src/cv_tool_interfaces`: custom ROS 2 action definition.

## Build and run with Docker (recommended)

### 1. Build image

From repository root:

```bash
docker build -t cv_tool:humble .
```

### 2. Run container

Basic run:

```bash
docker run --rm -it --net=host cv_tool:humble
```

Notes:
- `--net=host` is commonly needed for ROS 2 DDS discovery.
- If your camera is connected to the host and not published into ROS outside container, you may need extra device mounts/permissions.

### 3. What starts automatically

Container `CMD` launches:

```bash
ros2 launch cv_tool cv_tool.launch.py
```

The launch file currently starts the node with these defaults:
- `--rgb_topic /camera/camera/color/image_raw`
- `--depth_topic /camera/camera/depth/image_rect_raw`
- `--model 11n_int8`
- `--buffer_size 8`
- `--conf_thres 0.2`
- `--margin_x 70`
- `--margin_y 70`
- `--verbose`

## Build and run without Docker

If you want to run directly on a ROS 2 Humble host:

```bash
cd cv_tool_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Run node directly (lets you override arguments without editing launch):

```bash
ros2 run cv_tool cv_tool \
  --rgb_topic /camera/camera/color/image_raw \
  --depth_topic /camera/camera/depth/image_rect_raw \
  --model 11n_int8 \
  --buffer_size 8 \
  --conf_thres 0.2 \
  --margin_x 70 \
  --margin_y 70
```

## Parameters and how to change them

The node uses CLI arguments (not ROS parameter server parameters).
You can change values in two ways:

1. Edit `cv_tool_ws/src/cv_tool/launch/cv_tool.launch.py` in the `arguments=[...]` list.
2. Launch with `ros2 run cv_tool cv_tool ...` and pass desired flags.

### Supported arguments

- `--rgb_topic` (string)
  - Default: `/camera/camera/color/image_raw`
  - RGB image topic to subscribe.

- `--depth_topic` (string)
  - Default: `/camera/camera/depth/image_rect_raw`
  - Depth image topic to subscribe.

- `--model` (string)
  - Default: `11n_int8`
  - Selects model key used to resolve:
    - `/cv_tool_ws/src/cv_tool/cv_tool/models/<model>_openvino_model`

- `--buffer_size` (positive int)
  - Default: `8`
  - Number of consecutive frames where target must be detected before success.

- `--conf_thres` (float in `[0.1, 1.0]`)
  - Default: `0.2`
  - Confidence threshold for accepted detections.

- `--margin_x` (positive int)
  - Default: `70`
  - Horizontal centering tolerance in pixels.

- `--margin_y` (positive int)
  - Default: `70`
  - Vertical centering tolerance in pixels.

- `--verbose` (flag)
  - Default: off in node, on in provided launch file
  - Enables debug logging and saves annotated images to `output_images_<model>/`.

## Action API

Action: `cv_tool_interfaces/action/Detect`

Goal:
- `string tool_name`

Result:
- `bool success`
- `geometry_msgs/Point center`
- `geometry_msgs/Point top_left`
- `geometry_msgs/Point bottom_right`
- `float32 confidence`

Feedback:
- `string current_status`

### Example action call

```bash
ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect "{tool_name: screwdriver}" --feedback
```

Valid tool names currently expected by the server:
- `allen_small`
- `allen_large`
- `long_nose_pliers_large`
- `long_nose_pliers_small`
- `wire_stripper`
- `tape_measure`
- `cutting_pliers_large`
- `cutting_pliers_small`
- `combination_wrench`
- `multimeter`
- `screwdriver`
- `rachet`

## Using a custom model

The node loads OpenVINO models from this pattern:

- `/cv_tool_ws/src/cv_tool/cv_tool/models/<model_key>_openvino_model`

### A. Export your model to OpenVINO

Example with Ultralytics:

```bash
python3 -m pip install ultralytics openvino
python3 - << 'PY'
from ultralytics import YOLO
m = YOLO('best.pt')
m.export(format='openvino', int8=False)
PY
```

This produces an OpenVINO folder (typically containing `.xml` and `.bin`).

### B. Copy into package models directory

Example target:

```bash
cv_tool_ws/src/cv_tool/cv_tool/models/my_model_openvino_model/
```

Make sure folder contains model files expected by Ultralytics OpenVINO runtime (for example `best.xml` and `best.bin`).

### C. Enable your model key in argument parser

Edit `cv_tool_ws/src/cv_tool/cv_tool/cv_tool.py`, update `--model` choices to include your key, for example `my_model`.
Then run with:

```bash
ros2 run cv_tool cv_tool --model my_model
```

If you use launch file, also change `--model` value in `cv_tool.launch.py`.

### D. Rebuild

After code or model changes:

```bash
cd cv_tool_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

If using Docker image, rebuild image:

```bash
docker build -t cv_tool:humble .
```

## Important current note about model choices

In `cv_tool.py`, the parser currently allows model choices:
- `lll`
- `11l_int8`
- `11n`
- `11n_int8`

But repository model folders include `11l_openvino_model` (with `11l` naming).
If `11l` is intended, update parser choices accordingly before using it.
