# cv_tool (ROS 2 + Docker)

Action server for image-based tool detection used in the ARISE-KIRO project.
<!--- The node subscribes to RGB and depth camera topics, runs a custom YOLO model, and serves detection results through a ROS 2 action. -->

This ROS2 package detects industrial tools for robot picking. Implemented as an action server, it leverages Ultralytics YOLO and camera depth data to deproject 2D bounding boxes into real-world metric dimensions. After it receives a request, it continuously detects all tools in the camera's field of view and estimates their size until the target tool is identified. A detection is considered valid if the tool remains physically centered in the camera's view within a configurable pixel margin for a consecutive sequence of frames.

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

The launch file starts the node with a config file:
- `--config <package_share>/config/config.yaml`

Defaults live in `config.yaml`. Update that file and rebuild/re-source to apply changes inside the container.

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
ros2 run cv_tool cv_tool --config /cv_tool_ws/src/cv_tool/config/config.yaml
```

## Configuration

The node uses a YAML config file passed via `--config`.

Default config locations:
- Source tree: `cv_tool_ws/src/cv_tool/config/config.yaml`
- Installed: `/cv_tool_ws/install/cv_tool/share/cv_tool/config/config.yaml`

When running inside Docker, the launch file points at the installed config. Edit the source config and rebuild, or mount an external config and pass its absolute path.

### Supported arguments

- `--config` (string, required)
  - Path to `config.yaml` with all parameters.

### Config keys

- `rgb_topic` (string)
  - RGB image topic to subscribe.

- `depth_topic` (string)
  - Depth image topic to subscribe.

- `model_path` (string)
  - Path to OpenVINO model folder. Relative paths resolve under the package share
    (installed at `share/cv_tool/`), with a fallback to the Python module directory.

- `buffer_size` (positive int)
  - Number of consecutive frames where target must be detected before success.

- `conf_thres` (float in `[0.1, 1.0]`)
  - Confidence threshold for accepted detections.

- `margin_x` (positive int)
  - Horizontal centering tolerance in pixels.

- `margin_y` (positive int)
  - Vertical centering tolerance in pixels.

- `verbose` (bool)
  - Enables debug logging and saves annotated images to `output_images_<model>/`.

- `tool_class_names` (list of strings)
  - Valid tool names accepted by the action server.

- `camera_intrinsics` (mapping)
  - `fx`, `fy`, `cx`, `cy`, `depth_scale` used for 3D measurements.

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

Valid tool names are read from `tool_class_names` in the config. Default list:
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

The node loads YOLO, OpenVINO or compatible models from `model_path` in the config.
Relative paths are resolved under the package share (installed at `share/cv_tool/`).

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

Example target (relative path in config):

```bash
cv_tool_ws/src/cv_tool/cv_tool/models/my_model_openvino_model/
```

Make sure folder contains model files expected by Ultralytics OpenVINO runtime (for example `best.xml` and `best.bin`).

### C. Update config

Set in `config.yaml`:

```yaml
model_path: models/my_model_openvino_model
```

If you mount models from a host path in Docker, use an absolute path instead.

Note: If your custom model includes tools that look identical but differ in size, you may want to adjust the size heuristics in [cv_tool/cv_tool_ws/src/cv_tool/cv_tool/cv_tool.py](cv_tool/cv_tool_ws/src/cv_tool/cv_tool/cv_tool.py#L190) and [cv_tool/cv_tool_ws/src/cv_tool/cv_tool/cv_tool.py](cv_tool/cv_tool_ws/src/cv_tool/cv_tool/cv_tool.py#L212).

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

