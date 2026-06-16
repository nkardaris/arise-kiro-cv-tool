# cv_tool — Industrial Tool Recognition for Robotic Picking (ARISE-KIRO)

![Vulcanexus](https://img.shields.io/badge/vulcanexus-humble-4b4bcb)
![Humble](https://img.shields.io/badge/ros2-humble-0b4d91)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)

`cv_tool` is the **Tool recognition** reusable module of the **KIRO** experiment (ARISE 1st Open
Call). It is a ROS 2 / Vulcanexus **action server** that detects industrial tools for robotic
picking: after receiving a request for a named tool, it runs an Ultralytics YOLO model
(OpenVINO, CPU) on a live RGB-D stream and uses depth + camera intrinsics to deproject the 2D
bounding box into **3D metric coordinates**. A detection is accepted only when the tool stays
physically centered in the camera view within a configurable pixel margin for a consecutive run
of frames, so the robot only attempts to pick stable, well-positioned objects. Depth-based size
estimation also disambiguates visually identical tools of different scales (e.g. a small vs large
allen key) even when the model emits a generic class.

- **Inputs:** aligned RGB (`bgr8`) and depth (`16UC1`, millimetres) image topics; an action goal
  with the target `tool_name`.
- **Outputs:** the tool's 3D `center`, `top_left` and `bottom_right` points (camera color optical
  frame) and a `confidence` score, returned through a ROS 2 action.
- **Capability delivered:** off-the-shelf, real-time (≈15 fps CPU) tool detection + 3D
  localization for manipulation, with size disambiguation and centering-stability gating.

> **New here?** Read this README top-to-bottom (≈10 min), then the detailed pages under
> [`docs/`](docs/). The quickest check that everything installed is the
> [hello world](#hello-world-minimal-no-hardware) (no camera, no bag); to see real detections, run
> the [demo](#demo-sample-rosbag).

---

## Connection with ARISE

This module is the open implementation of the KIRO **`tool_recognition`** skill. In the KIRO
TRL6-7 demonstrator it is invoked by the mission-controller state machine during the
`pick_up_tool` task (`SearchToolState`): the manipulator sweeps the camera over the tool trays
while `cv_tool` scans for the requested tool; on a stable, centered detection the action succeeds,
the arm stops over the tool, and the returned bounding-box keypoints feed the downstream grasp
step (AprilTag ↔ tool association + vacuum pick).

- **ROS 2 / Vulcanexus:** the module runs on **Vulcanexus Humble** and exposes its capability
  purely over standard ROS 2 interfaces (one action + two image subscriptions over Fast DDS).
  See [`docs/02_interfaces.md`](docs/02_interfaces.md).
- **FIWARE / NGSI-LD, DDS enabler, ROS4HRI:** **not applicable to this module** — and that is by
  design, not an omission. In KIRO, FIWARE Orion-LD / NGSI-LD and the eProsima DDS↔NGSI-LD enabler
  operate centrally over the Robot/Mission/Worker entities, while this perception service talks to
  the mission controller synchronously over a ROS 2 action; ROS4HRI is used by KIRO's *human*
  modules (handover, human detection), not by tool detection. The full justification (and a
  candidate NGSI-LD mapping) is in [`docs/02_interfaces.md`](docs/02_interfaces.md#arise-middleware-interfaces--applicability).

This work is part of KIRO, co-funded by the European Union under the Horizon Europe **ARISE**
project (Grant Agreement No. 101135784).

## Target platforms

| Category | Tested on | Expected compatibility | Not supported / unknown |
|---|---|---|---|
| Manipulator / cobot | Universal Robots **UR10e** (mobile-base mounted) | Any arm — the module is platform-agnostic and only consumes camera topics | — |
| Sensor (RGB-D) | **Intel RealSense** depth camera (RealSense ROS 2 driver topics) | Any ROS 2 RGB-D source publishing aligned `bgr8` + `16UC1` depth with known intrinsics | Mono / RGB-only cameras (no 3D output, size disambiguation degraded) |
| Compute | x86-64 CPU (OpenVINO) | Any CPU supported by OpenVINO; GPU optional via Ultralytics | — |
| Runtime | Docker (Vulcanexus Humble image) | Native ROS 2 Humble host | ROS 1 |
| Replay (no hardware) | rosbag2 (sqlite3) RealSense recording | Any equivalent RGB-D bag/simulator | — |

## Robot missions and tasks

| | Description |
|---|---|
| **Mission** | On-demand **tool delivery / operator assistance** — a mobile manipulator retrieves tools for shop-floor workers. |
| **Task** | **Tool identification + 3D localization for picking**: given a requested tool id, find it among the tools in view and return its 3D keypoints for grasping. |
| **Skill provided** | `tool_recognition` (computer-vision detection + depth deprojection). |

## Off-the-shelf capabilities

| Capability | Input | Output | Interface | Status |
|---|---|---|---|---|
| Tool detection | RGB image | class + 2D bbox + confidence | `cv_tool_interfaces/action/Detect` | Implemented / tested |
| 3D localization | RGB + depth + intrinsics | 3D `center` / `top_left` / `bottom_right` (camera frame) | action result | Implemented / tested |
| Size disambiguation | depth bbox patch | small vs large class resolution (e.g. allen key) | internal heuristic | Implemented / tested |
| Centering-stability gating | bbox over N frames | accept only stable, centered detections | `buffer_size`, `margin_x/y` params | Implemented / tested |

**Measured performance** (KIRO D3 validation): detection **98.01 % precision / 98.46 % recall /
99.23 % mAP**, **~30 ms/frame**; pilot at IKH **F1 96.88 % / mAP 97.35 %** over ~2,300 frames with
**20/20** tool queries detected; CPU-optimized OpenVINO inference at **15 fps** (KPI target > 95 %).

---

## Hello world (minimal, no hardware)

The hello world just confirms everything is **installed and running** — no camera, no rosbag. The
default Docker `CMD` launches the action server, which loads the YOLO/OpenVINO model and waits for
goals.

```bash
# 1) Build the Docker image (from repo root)
docker build -t cv_tool:humble .

# 2) Start the action server
docker run --rm -it --net=host cv_tool:humble
```

**Success looks like** the server logging, within a few seconds:

```
Logging console output to logs/<DATE>_<TIME>.log
[cv_tool_action_server]: Loading YOLO model from: /.../share/cv_tool/models/11n_int8_openvino_model
[cv_tool_action_server]: CVToolActionServer ready. Waiting for goals...
```

That confirms the dependencies (rclpy, OpenCV, Ultralytics/OpenVINO/Torch), the bundled model, and
the ROS 2 action server all load correctly. Stop with `Ctrl-C`.

> Optional: in a second shell, `ros2 action info /detect_tool -t` lists the server. Sending a goal
> now aborts with *"No camera feed available"* — that is **expected** (no camera/bag connected) and
> still proves the action interface responds. To see real detections, run the demo below.

## Demo (sample rosbag)

The demo replays a recorded RealSense RGB-D rosbag so you get **real detections without any
hardware**. The bag is an **external download** (raw RGB-D is too large for git) — get the link and
topic details from [`examples/bags/README.md`](cv_tool_ws/src/cv_tool/examples/bags/README.md).

```bash
# 1) Run the container, mounting the folder that holds the downloaded bag
docker run --rm -it --net=host \
    -v /absolute/path/to/rosbags:/cv_tool_ws/bags \
    cv_tool:humble bash

# 2) Inside the container — replay the bag + start the action server
source /cv_tool_ws/install/setup.bash
ros2 launch cv_tool cv_tool_demo.launch.py bag_path:=/cv_tool_ws/bags/boxes_0.db3

# 3) In a second shell into the same container, request a tool
docker exec -it <container> bash
source /cv_tool_ws/install/setup.bash
ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect \
    "{tool_name: screwdriver}" --feedback
```

Expected: `Scanning frame N for screwdriver...` feedback, then `screwdriver found and centered!`,
returning `success: true` with three 3D points and a confidence score. Full walkthrough and
scenarios: [`docs/04_basic_demo_how_to_use.md`](docs/04_basic_demo_how_to_use.md).

> If the downloaded bag has no `metadata.yaml`, run `ros2 bag reindex /cv_tool_ws/bags -s sqlite3`
> once before launching. The same `docker run --rm -it --net=host cv_tool:humble` also works against
> a **live RealSense camera** publishing the topics in
> [`config/config.yaml`](cv_tool_ws/src/cv_tool/config/config.yaml).

## Build and run without Docker

```bash
cd cv_tool_ws
source /opt/ros/humble/setup.bash      # or the Vulcanexus setup
colcon build --symlink-install
source install/setup.bash
ros2 run cv_tool cv_tool --config /cv_tool_ws/src/cv_tool/config/config.yaml
```

---

## Configuration

The node takes a single `--config` argument pointing at a YAML file. Defaults live in
[`config/config.yaml`](cv_tool_ws/src/cv_tool/config/config.yaml); the launch files pass the
installed copy at `share/cv_tool/config/config.yaml`.

| Key | Type | Meaning |
|---|---|---|
| `rgb_topic` | string | RGB image topic to subscribe (`bgr8`). |
| `depth_topic` | string | Depth image topic to subscribe (`16UC1`, mm). |
| `model_path` | string | OpenVINO model folder. Relative paths resolve under the package share, then the module dir. |
| `buffer_size` | int > 0 | Consecutive frames the target must be detected before success. |
| `conf_thres` | float [0.1, 1.0] | Confidence threshold for accepted detections. |
| `margin_x`, `margin_y` | int > 0 | Horizontal / vertical centering tolerance in pixels. |
| `verbose` | bool | Debug logging + saves annotated images to `output_images_<model>/`. |
| `tool_class_names` | list[str] | Valid tool names accepted by the action server. |
| `camera_intrinsics` | mapping | `fx, fy, cx, cy, depth_scale` used for 3D measurements. |

## Action API

Action type: `cv_tool_interfaces/action/Detect` on endpoint `/detect_tool` (node
`cv_tool_action_server`).

```
# Goal
string tool_name
---
# Result
bool success
geometry_msgs/Point center
geometry_msgs/Point top_left
geometry_msgs/Point bottom_right
float32 confidence
---
# Feedback
string current_status
```

`tool_name` corresponds to the WMS/Mission tool id and must be one of `tool_class_names` in the
config. Default list: `allen_small`, `allen_large`, `long_nose_pliers_large`,
`long_nose_pliers_small`, `wire_stripper`, `tape_measure`, `cutting_pliers_large`,
`cutting_pliers_small`, `combination_wrench`, `multimeter`, `screwdriver`, `rachet`.

Full interface tables (topics, QoS, parameters, launch files) and the FIWARE/DDS/ROS4HRI
applicability discussion: [`docs/02_interfaces.md`](docs/02_interfaces.md).

## Basic demo

Beyond the hello world, [`docs/04_basic_demo_how_to_use.md`](docs/04_basic_demo_how_to_use.md)
walks through requesting different tools, the small-vs-large size disambiguation, the centering
behaviour, and how to read the annotated `verbose` output images.

## Using a custom model

The node loads YOLO / OpenVINO models from `model_path`. To use your own:

1. **Export to OpenVINO** (Ultralytics): `YOLO('best.pt').export(format='openvino')` → produces a
   folder with `best.xml` + `best.bin`.
2. **Copy** it under `cv_tool_ws/src/cv_tool/cv_tool/models/<your_model>_openvino_model/`.
3. **Point the config** at it: `model_path: models/<your_model>_openvino_model`.
4. **Rebuild** (`colcon build` / `docker build`).

If your model includes tools that look identical but differ in size, adjust the size heuristics in
[cv_tool.py](cv_tool_ws/src/cv_tool/cv_tool/cv_tool.py) (`tool_in_frame`, around the `0.17 m`
width/height threshold).

## Repository layout

```
.
├── Dockerfile                  # Vulcanexus Humble image; builds the workspace, launches the node
├── LICENSE                     # GNU AGPL-3.0
├── README.md                   # this file
├── docs/                       # ARISE context, interfaces, install/hello-world, demo, demonstrator role
│   ├── 01_arise_context.md
│   ├── 02_interfaces.md
│   ├── 03_installation_and_hello_world.md
│   ├── 04_basic_demo_how_to_use.md
│   ├── 05_role_in_demonstrator.md
│   └── D4_report.md            # ARISE D4 written report (draft)
├── media/                      # architecture diagram, screenshots, video links
└── cv_tool_ws/src/
    ├── cv_tool/                # Python action server, launch files, config, model, examples
    │   ├── cv_tool/            # node + utils
    │   ├── config/config.yaml
    │   ├── launch/             # cv_tool.launch.py, cv_tool_demo.launch.py
    │   └── examples/bags/      # how to fetch + replay the demo rosbag (external download)
    └── cv_tool_interfaces/     # Detect.action definition
```

## Limitations

- **No-hardware demo requires an external rosbag download** (not bundled in git due to size).
- **Depth-dependent size disambiguation:** small-vs-large class resolution relies on a fixed metric
  threshold and reliable depth; noisy or out-of-range depth (≤0.2 m or >1.0 m) falls back to the
  class name. Intrinsics in the config must match the camera.
- **Tuned for the KIRO tool set / tray layout** at a near-top-down working distance; other tools,
  backgrounds or distances may need re-training and re-tuning of `conf_thres` / margins / size
  threshold.
- Detection runs CPU-only by default (OpenVINO) at 15 fps; higher rates need `imgsz` reduction or GPU.
- See [`docs/02_interfaces.md`](docs/02_interfaces.md) and the
  [D4 report](docs/D4_report.md) for the full openness/limitations discussion.

## Maintainer, contact & citation

- **Maintainer:** Nikos Kardaris ([@nkardaris](https://github.com/nkardaris)),
  <nick.kardaris@gmail.com>. Issues: <https://github.com/nkardaris/arise-kiro-cv-tool/issues>.
- **Project contacts (IKNOWHOW SA):** Maria Kampa <mkampa@iknowhow.com>,
  Angeliki Pilalitou <apilalitou@iknowhow.com>.
- **Acknowledgement:** developed in the KIRO experiment, co-funded by the European Union under the
  Horizon Europe ARISE project (GA 101135784). Demonstrator video: **TODO (add URL)**.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
