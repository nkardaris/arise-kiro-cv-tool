# 05 — Role in the TRL6-7 demonstrator

## Demonstrator at a glance

| Item | Value |
|---|---|
| Demonstrator | KIRO — *Demonstration of HRI-enabled solution at work* (ARISE D3) |
| Environment | IKH facilities, **ARISTOS assembly production area** (real, space-constrained factory floor), pilot conditions |
| Robot / platform | **UR10e** arm on a mobile base; pneumatic vacuum gripper (PowerPick 10); **Intel RealSense** RGB-D camera on a 3D-printed mount |
| End user / scenario | Shop-floor operators requesting on-demand tool delivery during assembly |
| System TRL | 6 (this module ≈ TRL 5 as an extracted reusable asset) |
| Demonstrator video | **TODO (add URL)** |

## Problem the module addresses

During tool delivery, the robot must find a specific requested tool among many tools laid out in
trays/boxes, and localize it precisely enough to grasp it. Two complications: tools that look
identical but differ in size (e.g. small vs large allen key), and the need to only attempt a pick
when the tool is stable and well-positioned. `cv_tool` solves both — class detection plus
depth-based metric size disambiguation, gated by a centering-stability check.

## Module role in the full pipeline

The mission controller (YASMIN FSM) reaches `SearchToolState` for the `pick_up_tool` task:

1. The arm sweeps the camera across the tool trays along a Cartesian path.
2. In parallel, the controller opens a `Detect` action goal for the requested tool id.
3. `cv_tool` runs YOLO/OpenVINO on the RGB stream, deprojects the bbox using depth + intrinsics,
   and accepts a detection only once it is centered for `buffer_size` frames.
4. On success, the controller **cancels** the arm-sweep action (the end-effector stops over the
   tool) and transitions to `GrabToolState`.
5. `GrabToolState` associates the nearest AprilTag with the returned bounding-box centre and
   executes the vacuum pick.

So this module is the **perception trigger** that converts "keep sweeping" into "stop here and
grasp", and supplies the 3D keypoints the grasp step needs.

## What was extracted as reusable vs what stays demonstrator-specific

| Demonstrator component | Reusable here (`cv_tool`) | Stays demonstrator-specific |
|---|---|---|
| Tool perception | ✅ YOLO/OpenVINO detection, depth deprojection, size disambiguation, centering gating, `Detect` action | — |
| Arm sweep / motion | — | MoveIt2 Cartesian sweep + custom arm-trajectory action server |
| Grasping | — | AprilTag association, controller switching, vacuum gripper I/O |
| Mission orchestration | — | YASMIN mission controller, `/intents`, FIWARE/UWB/WMS integration |
| Camera bring-up | — | RealSense driver + 3D-printed mount + arm calibration |

## Validation evidence

From the KIRO D3 pilot:

- Detection (validation set): **98.01 % precision, 98.46 % recall, 99.23 % mAP, ~30 ms/frame**.
- Pilot at IKH: **F1 96.88 %, mAP 97.35 %** over ~2,300 active frames; **20/20** tool queries
  (action goals) detected; CPU OpenVINO at **15 fps**. KPI target was > 95 %.
- The module differentiated visually identical tools of different size during live operation.

Annotated example frames: [`../media/screenshots/`](../media/screenshots/). Demonstrator video:
see the table above (URL to be added).
