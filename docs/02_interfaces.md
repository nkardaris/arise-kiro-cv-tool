# 02 — Interfaces

## ROS 2 / Vulcanexus interface

Runtime: **Vulcanexus Humble** (ROS 2 Humble + Fast DDS). The module is a single node exposing one
action and consuming two image topics.

### Node

| Element | Name | Type | Description |
|---|---|---|---|
| Node | `cv_tool_action_server` | ROS 2 node (`rclpy`) | Loads the YOLO/OpenVINO model and serves the detection action. Uses a `ReentrantCallbackGroup` + `MultiThreadedExecutor`. |

### Action

| Element | Name | Type | Description |
|---|---|---|---|
| Action | `/detect_tool` | `cv_tool_interfaces/action/Detect` | Detect and 3D-localize a requested tool. |

```
# Goal
string tool_name            # requested tool id (must be in tool_class_names)
---
# Result
bool success
geometry_msgs/Point center        # 3D centroid of the bbox, camera color optical frame (m)
geometry_msgs/Point top_left      # 3D top-left bbox corner (m)
geometry_msgs/Point bottom_right  # 3D bottom-right bbox corner (m)
float32 confidence
---
# Feedback
string current_status       # e.g. "Scanning frame N for screwdriver..."
```

**Behaviour / outcomes**

- Goal **aborted** (`success: false`) if `tool_name` is not in `tool_class_names`, or if no RGB
  frame has been received yet.
- If no depth frame is available, the server proceeds **RGB-only** with a warning; 3D points and
  size disambiguation are then unreliable.
- The goal **succeeds** once the target is detected, `conf > conf_thres`, and **centered** within
  `margin_x`/`margin_y` for `buffer_size` consecutive frames. Processing is throttled to ~15 Hz.
- 3D points are deprojected with the pinhole model using `camera_intrinsics` and the median depth
  over the bounding-box patch.

### Subscriptions

| Element | Name (default) | Type | QoS | Description |
|---|---|---|---|---|
| Subscribe | `rgb_topic` = `/camera/camera/color/image_raw` | `sensor_msgs/Image` (`bgr8`) | depth 10 (default reliable) | RGB frames for YOLO inference. |
| Subscribe | `depth_topic` = `/camera/camera/depth/image_rect_raw` | `sensor_msgs/Image` (`16UC1`, mm) | depth 10 | Aligned depth for 3D deprojection and size estimation. |

The module publishes no topics; results are returned via the action.

### Parameters (config file)

Parameters are provided through a YAML config passed with `--config`, not as ROS parameters. See
[`config/config.yaml`](../cv_tool_ws/src/cv_tool/config/config.yaml) and the parameter table in the
[README](../README.md#configuration). Key items: `model_path`, `buffer_size`, `conf_thres`,
`margin_x`, `margin_y`, `tool_class_names`, `camera_intrinsics` (`fx, fy, cx, cy, depth_scale`).

### Launch files

| File | Purpose |
|---|---|
| [`launch/cv_tool.launch.py`](../cv_tool_ws/src/cv_tool/launch/cv_tool.launch.py) | Start the action server with the installed config (live-camera / full demo). |
| [`launch/cv_tool_demo.launch.py`](../cv_tool_ws/src/cv_tool/launch/cv_tool_demo.launch.py) | Replay the demo rosbag (`bag_path` arg) **and** start the server — the hardware-free demo. |

---

## ARISE middleware interfaces — applicability

The minimum ARISE interfaces are ROS 2/Vulcanexus, FIWARE/NGSI-LD, the DDS↔NGSI-LD enabler, and
ROS4HRI. For this module only ROS 2/Vulcanexus applies; the rest are **N/A by design**, justified
below per the D4 guidance (§3.2.6).

### FIWARE / NGSI-LD — N/A (handled centrally in KIRO)

`cv_tool` is an in-cell perception service that returns results **synchronously** to the mission
controller over a ROS 2 action. It does not publish state to a context broker. In the KIRO
architecture (D3 §1.2.3), FIWARE Orion-LD / NGSI-LD integration is centralized: the broker holds the
`Robot`, `Mission` and `Worker` entities and the eProsima DDS↔NGSI-LD enabler bridges those to ROS 2
topics. The tool request reaches this module indirectly (Mission `tool.id` → `/mission/tool_id` →
`/intents` → mission controller → `Detect` goal). Adding a per-module broker dependency would
duplicate that central integration and couple a real-time perception node to the broker.

**Candidate NGSI-LD mapping (for reference, not implemented).** If a future deployment needed the
detection result on the broker, the action result maps cleanly to a `ToolDetection` entity:

```json
{
  "id": "urn:ngsi-ld:ToolDetection:cv_tool:screwdriver",
  "type": "ToolDetection",
  "toolName":   { "type": "Property", "value": "screwdriver" },
  "confidence": { "type": "Property", "value": 0.93 },
  "center":     { "type": "Property", "value": {"x": 0.01, "y": -0.02, "z": 0.48}, "unitCode": "MTR" },
  "topLeft":    { "type": "Property", "value": {"x": -0.05, "y": -0.06, "z": 0.48}, "unitCode": "MTR" },
  "bottomRight":{ "type": "Property", "value": {"x":  0.06, "y":  0.05, "z": 0.48}, "unitCode": "MTR" },
  "refRobot":   { "type": "Relationship", "object": "urn:ngsi-ld:Robot:kiro" },
  "@context": [
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
  ]
}
```

### DDS↔NGSI-LD enabler — N/A here

The eProsima DDS Router + Fast DDS Discovery Server (Vulcanexus Jazzy container) act as KIRO's
DDS↔NGSI-LD enabler, forwarding only the entity topics listed in the D3 integration table. The
`/detect_tool` action is intentionally **not** part of that bridge, so no enabler configuration file
is shipped in this repository; the enabler configuration lives at the KIRO system level.

### ROS4HRI / ROS4RI — N/A (no human perception)

This module perceives **tools, not humans**, so it neither produces nor consumes ROS4HRI concepts
(bodies, faces, voices, skeletons, intents). ROS4HRI is applied elsewhere in KIRO — the handover
module publishes/consumes `/humans/bodies/<id>/...` and the human-detection module publishes
`/tracked_agents`. A plausible **future** alignment would be to present a detected tool toward a
ROS4HRI-tracked operator (linking a `ToolDetection` to a `/humans/bodies/<id>`), but that belongs to
the handover pipeline, not to tool recognition.
