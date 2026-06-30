# 01 — ARISE context

## What this module is

`cv_tool` is the **Tool recognition** reusable module produced by the **KIRO** experiment (*Key
Intelligent & Interactive Robotic Operator*), funded under the ARISE 1st Open Call (Horizon Europe,
GA 101135784, lead **IKNOWHOW SA**). It packages, as open and independently runnable software, one
software module of the KIRO TRL6-7 demonstrator: a ROS 2 / Vulcanexus **action server** that
recognizes industrial tools and returns their 3D position for robotic picking.

It is listed as a reusable module in the KIRO D3 deliverable:

## KIRO in one paragraph

KIRO is an HRI-enabled mobile manipulator (UR10e arm on a mobile base) that delivers tools to
shop-floor operators on demand. A worker requests a tool by voice/app; an LLM agent and the FIWARE
Orion-LD context broker resolve the request into a mission; a ROS 2 mission controller (a YASMIN
state machine) then drives human-aware navigation, **tool recognition + picking**, and a socially
/ ergonomically aware handover. KIRO reached **TRL 6** in a pilot at IKH's ARISTOS assembly area.

## Where this module sits in the demonstrator

The mission controller decomposes a delivery mission into tasks and skills. `cv_tool` provides the
**`tool_recognition`** skill, used by the **`pick_up_tool`** task:

```
user voice/app request
        │  (FIWARE Orion-LD → /mission/tool_id → /intents)
        ▼
kiro_mission_controller (FSM)
        │
   task: pick_up_tool ──► SearchToolState
        │                     │  action client → /detect_tool  (THIS MODULE)
        │                     │  goal: requested tool id
        │                     ▼
        │              cv_tool_action_server  ── scans RGB-D, returns 3D bbox keypoints
        ▼
   GrabToolState (AprilTag ↔ tool, vacuum pick)
```

In `SearchToolState`, while the arm sweeps the camera over the tool trays, the mission controller
opens a `Detect` action goal for the requested tool. When `cv_tool` reports a stable, centered
detection, the action succeeds, the arm-motion action is cancelled (the end-effector stops over the
tool), and the returned bounding-box keypoints are handed to `GrabToolState` for the vacuum pick.

## ARISE middleware alignment (summary)

| Concern | This module |
|---|---|
| **ROS 2 / Vulcanexus** | ✅ Core interface. Vulcanexus Humble; one ROS 2 action + two image subscriptions over Fast DDS. |
| **FIWARE / NGSI-LD** | N/A for this module — handled centrally in KIRO (Mission/Robot/Worker entities), not by perception. See [`02_interfaces.md`](02_interfaces.md#arise-middleware-interfaces--applicability). |
| **DDS↔NGSI-LD enabler** | N/A here — the eProsima enabler bridges the entity topics at system level, not the detection action. |
| **ROS4HRI / ROS4RI** | N/A — the module perceives *tools*, not humans. ROS4HRI is used by KIRO's handover & human-detection modules. |

## What is open here

The complete detection capability is open: the ROS 2 node, the `Detect` action definition, the
launch files, the configuration, and a bundled YOLO/OpenVINO model. The hardware-free execution
path (recorded RealSense rosbag replay) is documented and reproducible. What remains
demonstrator-specific (mission controller, grasping/AprilTag logic, the robot/gripper drivers) is
**not** part of this module and is described in [`05_role_in_demonstrator.md`](05_role_in_demonstrator.md).
