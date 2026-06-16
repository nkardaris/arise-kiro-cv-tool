#!/usr/bin/env python3
"""Fill Chapter 3 of the ARISE D4 template with the KIRO cv_tool content, in place.

Reads  docs/D4 - Shareable HRI Modules.docx  (the blank ARISE template) and writes
       docs/D4 - Shareable HRI Modules - KIRO cv_tool.docx  (the filled submission)
preserving the template's native styles, fonts and branding.

Regenerate with:  python3 docs/_fill_template_docx.py   (requires python-docx)
"""
import os
from docx import Document

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "D4 - Shareable HRI Modules.docx")
DST = os.path.join(HERE, "D4 - Shareable HRI Modules - KIRO cv_tool.docx")

doc = Document(SRC)
T = doc.tables  # T[7..41] are the Chapter-3 fillable tables (see _map)


# --------------------------------------------------------------------------- helpers
def set_cell(cell, text):
    """Set cell text, preserving the first run's formatting (font/size)."""
    text = "" if text is None else str(text)
    p = cell.paragraphs[0]
    # drop extra paragraphs
    for extra in cell.paragraphs[1:]:
        extra._element.getparent().remove(extra._element)
    runs = p.runs
    if runs:
        first = runs[0]
        for r in runs[1:]:
            r._element.getparent().remove(r._element)
        if "\n" in text:
            lines = text.split("\n")
            first.text = lines[0]
            for ln in lines[1:]:
                first.add_break()
                first.add_text(ln)
        else:
            first.text = text
    else:
        p.add_run(text)


def fill(table, rows, start=1, right_only=False):
    """Fill data rows. If right_only, only set the last column (key/value tables)."""
    for i, rowvals in enumerate(rows):
        ri = start + i
        if ri >= len(table.rows):
            cells = table.add_row().cells
        else:
            cells = table.rows[ri].cells
        if right_only:
            set_cell(cells[-1], rowvals)
        else:
            for ci, val in enumerate(rowvals):
                if ci < len(cells):
                    set_cell(cells[ci], val)


def blank_rest(table, after):
    """Replace any leftover template placeholder rows with '—'."""
    for ri in range(after + 1, len(table.rows)):
        for c in table.rows[ri].cells:
            if c.text.strip().startswith("[") or c.text.strip() == "":
                set_cell(c, "—")


def set_para_text(contains, new_text):
    """Replace the text of the first paragraph containing `contains`."""
    for p in doc.paragraphs:
        if contains in p.text:
            if p.runs:
                p.runs[0].text = new_text
                for r in p.runs[1:]:
                    r._element.getparent().remove(r._element)
            else:
                p.add_run(new_text)
            return True
    return False


def tick_checklists():
    keywords = ("readme", "license", "repository", "ros 2", "fiware", "dds", "ros4hri",
                "installation", "hello world", "role in", "known", "maintainer",
                "project identification", "implementation", "simulation path", "limitations",
                "annexes", "trl6-7", "basic demo")
    for p in doc.paragraphs:
        t = p.text
        tl = t.lower()
        if "[ ]" in t and any(k in tl for k in keywords):
            keep_open = ("video" in tl) or ("screenshots" in tl) or ("visual evidence" in tl)
            if not keep_open and p.runs:
                for r in p.runs:
                    if "[ ]" in r.text:
                        r.text = r.text.replace("[ ]", "[x]")


# =========================================================================== 3.1
fill(T[7], [
    "Key Intelligent & Interactive Robotic Operator",
    "KIRO",
    "IKNOWHOW SA (IKH)",
    "Maria Kampa, mkampa@iknowhow.com, IKH; Angeliki Pilalitou, apilalitou@iknowhow.com, IKH",
    "Nikos Kardaris, nick.kardaris@gmail.com, IKNOWHOW SA, GitHub @nkardaris",
    "Politecnico di Milano — Mostafa Zarei (mostafa.zarei@polimi.it), Amirshayan Nasirimajd, "
    "Nima Rahmani, Walter Quadrini",
    "cv_tool (Tool recognition)",
    "https://github.com/nkardaris/arise-kiro-cv-tool",
    "TODO: v0.1.0 (tag after merge)",
    "TODO: demonstrator video URL",
], right_only=True)

# =========================================================================== 3.2.1
fill(T[8], [
    "arise-kiro-cv-tool", "nkardaris", "https://github.com/nkardaris/arise-kiro-cv-tool",
    "TODO: Public", "TODO: v0.1.0", "main", "Python",
    "ROS 2 / Vulcanexus Humble (Docker)", "Yes (GitHub Issues)",
], right_only=True)
fill(T[9], [
    "Confirmed — ARISE reviewers can access the repository, documentation, release assets and "
    "the rosbag download link.",
    "TODO: public now / at submission.",
    "Demo rosbag boxes_0.db3 (external download); pip ML deps (Ultralytics, OpenVINO, PyTorch) "
    "pinned in the Dockerfile.",
    "Confirmed — the submitted release/tag will not be modified during review (stable until "
    "30.06.2027).",
], right_only=True)

# =========================================================================== 3.2.2
fill(T[10], [
    "GNU AGPL-3.0-or-later (see LICENSE in the repository root).",
    "IKNOWHOW SA / KIRO.",
    "Ultralytics YOLO (AGPL-3.0), OpenVINO (Apache-2.0), PyTorch (BSD), OpenCV (Apache-2.0), "
    "ROS 2 / Vulcanexus packages.",
    "Nikos Kardaris, IKNOWHOW SA, nick.kardaris@gmail.com, @nkardaris.",
    "Best effort through project end.",
    "The entire cv_tool module is open; KIRO's RaaS / module-licensing offering is separate and "
    "out of scope for this module.",
], right_only=True)

# =========================================================================== 3.2.3
fill(T[11], [
    ["Tool detection + 3D localization", "Open", "cv_tool_ws/src/cv_tool/cv_tool/, launch/, config/",
     "Full capability: YOLO/OpenVINO + depth deprojection + centering gating"],
    ["Bundled detection model", "Open", "cv_tool/models/11n_int8_openvino_model/",
     "Trained on the KIRO tool set"],
    ["Camera interface (RGB-D)", "Open driver config / N/A driver",
     "config/config.yaml (topics + intrinsics)",
     "Intel RealSense ROS 2 driver topics; driver itself is upstream"],
    ["Grasping / mission orchestration", "Excluded (demonstrator-specific)", "—",
     "AprilTag grasp, YASMIN mission controller, FIWARE/UWB/WMS — out of scope"],
])

# =========================================================================== 3.2.5 platforms
fill(T[15], [
    ["Manipulator/cobot", "UR10e (mobile-base mounted)", "Any arm (platform-agnostic)", "—"],
    ["Mobile robot / AMR / AGV", "KIRO mobile base", "Any", "—"],
    ["Humanoid/social robot", "—", "—", "Out of scope"],
    ["Industrial cell or PLC-integrated setup", "IKH ARISTOS assembly area (pilot)", "Similar cells", "—"],
    ["Sensors", "Intel RealSense RGB-D", "Any aligned bgr8 + 16UC1 RGB-D w/ known intrinsics", "Mono / RGB-only"],
    ["Simulation", "rosbag2 (recorded RealSense)", "Equivalent bag / simulator", "—"],
])

# =========================================================================== 3.2.6 ROS2 interface
fill(T[16], [
    ["Node", "/cv_tool_action_server", "ROS 2 node (rclpy)",
     "Loads the YOLO/OpenVINO model and serves the detection action"],
    ["Subscribes", "/camera/camera/color/image_raw", "sensor_msgs/Image (bgr8)",
     "RGB frames for inference (QoS depth 10)"],
    ["Subscribes", "/camera/camera/depth/image_rect_raw", "sensor_msgs/Image (16UC1, mm)",
     "Aligned depth for 3D deprojection (QoS depth 10)"],
    ["Action", "/detect_tool", "cv_tool_interfaces/action/Detect",
     "Goal: string tool_name → detect + 3D-localize"],
    ["Result", "success/center/top_left/bottom_right/confidence", "bool / geometry_msgs/Point ×3 / float32",
     "3D keypoints (camera color optical frame) + confidence"],
    ["Feedback", "current_status", "string", "Scan progress, e.g. 'Scanning frame N...'"],
    ["Launch file", "cv_tool.launch.py / cv_tool_replay.launch.py", "launch",
     "Live camera (full demo) / rosbag replay (hello world)"],
])

# =========================================================================== 3.2.6 FIWARE entity table
fill(T[17], [
    ["ToolDetection", "toolName (Property)", "Detected tool id (string); candidate mapping only — not implemented"],
    ["ToolDetection", "confidence (Property)", "Detection confidence in [0,1]"],
    ["ToolDetection", "center / topLeft / bottomRight (Property)", "3D points (m, unitCode MTR), camera frame; refRobot Relationship → urn:ngsi-ld:Robot:kiro"],
])
# 3.2.6 candidate JSON (single-cell code table)
set_cell(T[18].rows[0].cells[0],
    '{\n'
    '  "id": "urn:ngsi-ld:ToolDetection:cv_tool:screwdriver",\n'
    '  "type": "ToolDetection",\n'
    '  "toolName":   { "type": "Property", "value": "screwdriver" },\n'
    '  "confidence": { "type": "Property", "value": 0.93 },\n'
    '  "center":     { "type": "Property", "value": {"x":0.01,"y":-0.02,"z":0.48}, "unitCode": "MTR" },\n'
    '  "refRobot":   { "type": "Relationship", "object": "urn:ngsi-ld:Robot:kiro" },\n'
    '  "@context": [ "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld" ]\n'
    '}\n'
    '// Candidate mapping only — NGSI-LD is handled centrally in KIRO, not by this module.')

# =========================================================================== 3.2.6 DDS enabler table
fill(T[19], [
    ["DDS enabler configuration file", "—", "N/A — central enabler at system level"],
    ["Mapped DDS topics or types", "docs/02_interfaces.md", "N/A for this module"],
    ["Mapped NGSI-LD entities", "docs/02_interfaces.md", "N/A (candidate ToolDetection mapping only)"],
    ["Test command or script", "—", "N/A"],
    ["Known limitations", "docs/02_interfaces.md", "Action-based; not bridged to NGSI-LD"],
])

# =========================================================================== 3.2.6 ROS4HRI table
fill(T[20], [
    ["Human presence", "—", "N/A", "Module perceives tools, not humans"],
    ["Gesture or action", "—", "N/A", "ROS4HRI used by KIRO handover / human-detection modules"],
    ["Operator state", "—", "N/A", "—"],
    ["Speech or intent", "—", "N/A", "Tool id arrives via mission controller, not ROS4HRI"],
])

# =========================================================================== 3.2.7 deps
fill(T[21], [
    ["Operating system", "Ubuntu 22.04 (image)", "Ubuntu 22.04", "Dockerfile"],
    ["ROS 2 / Vulcanexus", "Vulcanexus Humble", "Vulcanexus Humble", "Dockerfile"],
    ["Python/C++ dependencies", "ultralytics, openvino, torch (CPU), numpy<2, lap", "same",
     "Dockerfile / package.xml"],
    ["Docker", "Yes", "Yes", "Dockerfile"],
    ["FIWARE / Context Broker", "N/A", "N/A", "docs/02_interfaces.md"],
    ["Hardware", "None (recorded bag)", "Intel RealSense RGB-D", "examples/bags/README.md"],
    ["Simulation or recorded data", "rosbag2 (RealSense recording)", "live camera",
     "examples/bags/README.md"],
])

# =========================================================================== 3.3.1 exec summary
fill(T[22], [
    "A ROS 2 / Vulcanexus action server that detects a requested industrial tool in an RGB-D "
    "stream (Ultralytics YOLO on OpenVINO), deprojects the 2D bounding box into 3D metric "
    "coordinates using depth + camera intrinsics, and returns the tool's 3D center/top-left/"
    "bottom-right points and confidence. It accepts a detection only when the tool is centered "
    "and stable across consecutive frames, and disambiguates visually identical tools of "
    "different size from their measured dimensions.",
    "Finding and precisely localizing a specific requested tool among many in trays/boxes during "
    "robotic tool delivery — including size-ambiguous tools — so a manipulator can grasp it.",
    "Real-time (~15 fps CPU) tool detection + 3D localization via one ROS 2 action; size "
    "disambiguation; centering-stability gating; a bundled model and a no-hardware rosbag demo.",
    "Robotics developers / integrators building pick-and-deliver or bin-picking pipelines on "
    "ROS 2 / Vulcanexus; the Vulcanexus community; ARISE ecosystem adopters.",
    "Docker (Vulcanexus Humble) + recorded RealSense rosbag replay — no camera or robot required.",
    "KIRO D3 pilot at IKH: F1 96.88%, mAP 97.35%, 20/20 tool queries detected; demonstrator "
    "video (TODO).",
], right_only=True)

# =========================================================================== 3.3.2 milestones
fill(T[23], [
    ["Stage 1 - Individual Mentoring Plan", "Need: robot tool delivery; CV to identify/pick tools",
     "Defined the tool-recognition scope"],
    ["Stage 2 - Proof of Concept", "Tool recognition MVP; UR10e + RealSense integration",
     "Detection + 3D localization approach retained"],
    ["Stage 3 - TRL6-7 Demonstrator", "Validated in the IKH pilot (F1 96.88%, mAP 97.35%, 20/20)",
     "This action server is the extracted, validated capability"],
    ["Stage 4 - Shareable module", "Open implementation + documentation + hardware-free demo",
     "Now reusable by third parties"],
])
set_para_text("ARISE alignment narrative",
    "ARISE alignment: the module runs on Vulcanexus Humble and exposes its capability over "
    "standard ROS 2 (one action over Fast DDS). FIWARE/NGSI-LD and the DDS↔NGSI-LD enabler are "
    "handled centrally in KIRO over the Robot/Mission/Worker entities, not by this perception "
    "node; ROS4HRI is applied by KIRO's human-facing modules (handover, human detection). The "
    "applicability of each interface is justified in docs/02_interfaces.md, to be confirmed with "
    "the ARISE mentor.")

# =========================================================================== 3.3.3 platforms/missions/tasks
fill(T[24], [
    ["Manipulator/cobot", "Yes - UR10e", "Any arm", "Needs an RGB-D camera in the work area"],
    ["Mobile robot / AMR / AGV", "Yes - KIRO base", "Any", "—"],
    ["Humanoid/social robot", "No", "—", "Out of scope"],
    ["Industrial cell or production environment", "Yes - IKH ARISTOS", "Similar cells",
     "Tuned to the KIRO tool set / tray layout"],
    ["Sensor stack", "Yes - Intel RealSense RGB-D", "Any aligned bgr8+16UC1 RGB-D",
     "Mono/RGB-only loses 3D + size disambiguation"],
    ["Simulator or recorded data", "Yes - rosbag2", "Equivalent bag / sim", "—"],
])
fill(T[25], [
    ["Collaborative assembly", "Yes", "On-demand tool delivery to assembly operators"],
    ["Human-aware navigation", "No", "Handled by KIRO social navigation module"],
    ["Object handover", "No", "Handled by KIRO close-proximity HRI module"],
    ["Operator monitoring or assistance", "Yes", "Provides the tool the operator requested"],
    ["Quality inspection", "N/A", "—"],
    ["Intralogistics", "N/A", "—"],
    ["Teleoperation or remote supervision", "N/A", "—"],
    ["Safety-aware task execution", "Partial", "Only picks stable, centered detections"],
    ["Other", "—", "—"],
])
fill(T[26], [
    ["Tool identification + 3D localization for picking", "RGB-D + tool id (action goal)",
     "3D keypoints + confidence (action result)", "Implemented / tested"],
])
blank_rest(T[26], 1)

# =========================================================================== 3.3.4 capabilities
fill(T[27], [
    ["Tool detection (+ centering gating)", "RGB", "class + 2D bbox + confidence", "ROS 2 action",
     "Implemented / tested"],
    ["3D localization", "RGB + depth + intrinsics", "3D center/top_left/bottom_right (m)",
     "ROS 2 action", "Implemented / tested"],
    ["Size disambiguation", "depth bbox patch", "small vs large class (e.g. allen key)",
     "internal", "Implemented / tested"],
])

# =========================================================================== 3.3.5 interoperability
fill(T[28], [
    ["ROS 2/Vulcanexus", "Node, action, topics, QoS, params, launch files", "docs/02_interfaces.md"],
    ["FIWARE/NGSI-LD", "N/A justification + candidate ToolDetection mapping", "docs/02_interfaces.md"],
    ["DDS NGSI-LD mapping tool / enabler", "N/A — central enabler at system level", "docs/02_interfaces.md"],
    ["ROS4HRI/ROS4RI", "N/A — module perceives tools, not humans", "docs/02_interfaces.md"],
    ["Other relevant standard/interface", "YOLO/OpenVINO model; CPU inference", "cv_tool/models/, cv_tool.py"],
])

# =========================================================================== 3.3.6 Vulcanexus
fill(T[29], [
    "Vulcanexus Humble image (eprosima/vulcanexus:humble-desktop); ROS 2 Humble action server + "
    "image subscriptions; launch files; CPU OpenVINO inference.",
    "Fast DDS transport for the action and image streams; --net=host discovery; rosbag2 replay "
    "for hardware-free reproduction.",
    "A reusable, packaged ROS 2 HRI/perception capability (tool detection + 3D localization) with "
    "a documented Docker + rosbag demo; candidate for a Vulcanexus example/tutorial.",
    "Yes — self-contained; would need a publicly hosted demo bag link + a release tag.",
    "Repository, this report, annotated frames, D3 metrics; demonstrator video (TODO).",
    "Pip-only ML deps (Ultralytics/OpenVINO/PyTorch); the robot/camera drivers and grasping logic.",
], right_only=True)

# =========================================================================== 3.3.7 execution evidence
fill(T[30], [
    "Docker build of the Vulcanexus image; native colcon path also documented (docs/03).",
    "Replay the recorded RealSense rosbag + send a Detect goal; returns success:true with 3D "
    "points. No hardware needed.",
    "Different tools, size disambiguation, centering behaviour, verbose annotated output "
    "(docs/04).",
], right_only=True)
# rows 4-5 (Simulation/mock path, Troubleshooting) also key/value
set_cell(T[30].rows[4].cells[1],
    "Recorded RealSense rosbag (external download) replayed via cv_tool_replay.launch.py.")
set_cell(T[30].rows[5].cells[1],
    "Documented failure modes (no feed, missing bag metadata, DDS discovery) in docs/03.")
set_para_text("summarise the installation and execution evidence",
    "Installation and execution evidence: building the Docker image and replaying the recorded "
    "RealSense rosbag exercises the full pipeline without any hardware. With the bag mounted, "
    "'ros2 launch cv_tool cv_tool_replay.launch.py' plus a Detect goal yields 'Scanning frame N...' "
    "feedback and then a success result carrying the tool's 3D center/top-left/bottom-right points "
    "and confidence — reproducing the behaviour validated in the KIRO D3 pilot (F1 96.88%, mAP "
    "97.35%, 20/20 queries).")

# =========================================================================== 3.3.8 demonstrator
fill(T[31], [
    "KIRO D3 — Demonstration of HRI-enabled solution at work",
    "IKH facilities, ARISTOS assembly production area (real factory floor), pilot conditions",
    "UR10e arm + mobile base; Intel RealSense RGB-D; PowerPick 10 pneumatic vacuum gripper",
    "Shop-floor operators; on-demand tool delivery during assembly",
    "Find + localize a requested tool (incl. size-ambiguous ones) for robotic picking",
    "tool_recognition skill in SearchToolState (pick_up_tool task): triggers 'stop & grasp' and "
    "supplies the 3D bounding-box keypoints used by the downstream grasp step",
    "TODO: demonstrator video URL",
], right_only=True)
fill(T[32], [
    ["Tool perception (SearchToolState)", "YOLO/OpenVINO detection + depth deprojection + size "
     "disambiguation + Detect action", "—"],
    ["Arm sweep / motion", "—", "MoveIt2 Cartesian sweep + custom arm-trajectory action server"],
    ["Grasping (GrabToolState)", "—", "AprilTag association, controller switching, vacuum gripper I/O"],
])

# =========================================================================== 3.3.9 validation
fill(T[33], [
    ["Demonstrator video", "TODO", "End-to-end use case / module behaviour"],
    ["Screenshots or diagrams", "media/architecture_diagram.md; media/screenshots/ (add frames)",
     "System context + annotated detections"],
    ["Execution logs", "docs/03 expected output", "Hello-world success path"],
    ["Metrics", "F1 96.88%, mAP 97.35%, 20/20 queries, ~30 ms/frame, 15 fps", "Accuracy / success rate"],
    ["Impact on End-user (industrial added value)", "Reduced tool-fetching overhead; supports "
     "productivity and ergonomics", "Industrial relevance"],
    ["Impact on tech provider (Reuse potential)", "Reusable ROS 2 detection+localization block "
     "for pick/bin-picking pipelines", "Who could reuse/extend the module"],
])
set_para_text("provide a concise impact story",
    "Impact story: for the end user (IKH operators), the module underpins on-demand tool delivery "
    "that reduces interruptions and walking to fetch tools, improving workflow continuity and "
    "ergonomics. For the technology provider, it is a reusable, validated ROS 2/Vulcanexus block "
    "(detection + 3D localization with size disambiguation) that can be dropped into other "
    "pick-and-deliver or bin-picking pipelines, accelerating development versus building tool "
    "perception from scratch.")

# =========================================================================== 3.3.10 openness/limitations
fill(T[34], [
    "Node, Detect action, launch files, config, bundled model and the rosbag replay path — all "
    "open under AGPL-3.0.",
    "None withheld for this module; KIRO's RaaS / integration offering is separate.",
    "Depth-dependent size disambiguation (fixed ~0.17 m threshold; reliable depth 0.2–1.0 m); "
    "tuned to the KIRO tool set / tray layout at a near-top-down distance; CPU 15 fps.",
    "Needs an aligned RGB-D source with known intrinsics.",
    "Other tool sets / backgrounds / distances; non-RealSense cameras; GPU path.",
    "Broaden tool set / retraining; optional NGSI-LD publication; optional ROS4HRI pairing for "
    "tool-to-operator presentation.",
    "Perceives tools, not people; processes no personal data. KIRO ethics handled at system level.",
], right_only=True)

# =========================================================================== 3.3.11 self-assessment
fill(T[35], [
    ["Related project", "Yes", "Meets all minimum requirements; can be listed with a link."],
    ["Featured project", "Yes (target)", "Good reproducibility (Docker + rosbag), clear README + "
     "docs, strong validated metrics, clean ROS 2/Vulcanexus interface."],
    ["Flagship project", "No", "FIWARE/ROS4HRI are (justifiably) N/A for this perception module; "
     "public demo video/bag link pending."],
])
fill(T[36], [
    "Platform-agnostic action interface; only consumes camera topics.",
    "Docker image + hardware-free rosbag hello world with documented expected output.",
    "ROS 2/Vulcanexus core; FIWARE/DDS/ROS4HRI justified N/A.",
    "Packaged reusable HRI perception capability + Docker/rosbag demo; candidate Vulcanexus example.",
    "D3 pilot metrics + (TODO) demonstrator video.",
    "README + 5 docs pages + interface tables + architecture/sequence diagrams.",
    "AGPL-3.0, named maintainer, issue tracker, planned v0.1.0 tag.",
], right_only=True)

# =========================================================================== 3.4.1 Annex I
fill(T[37], [
    ["Stage 1 - Individual Mentoring Plan",
     "https://drive.google.com/file/d/14ZL7eP12LT9KzNtxIwyUyXYDJIsFaZLc/view",
     "Need/challenge/scope reused in D4"],
    ["Stage 2 - Proof of Concept deliverable",
     "https://drive.google.com/drive/folders/1wEI_4xPRaafFpAKw67d1bXBSl7aKGYz0",
     "Prototype capability reused in D4"],
    ["Stage 3 - Demonstrator deliverable (D2.2 MVP / D3)",
     "https://drive.google.com/drive/folders/11UyrMabCm_4WHTXDbXODwisRI4qAAOSK",
     "TRL6-7 evidence reused in D4"],
    ["Demonstrator video", "TODO", "Validation evidence"],
    ["Additional technical documentation",
     "D2.1 video: https://drive.google.com/drive/folders/1X5QbdzbXpFyobgnRyQieVTStlsA7-0w9",
     "Interfaces, datasets, hardware"],
])
fill(T[38], [
    ["4", "TODO: copy the recommendation(s) received after the D3 evaluation, then reply here."],
    ["4", "TODO: copy the recommendation(s) received after the D3 evaluation, then reply here."],
])

# =========================================================================== 3.4.2 Annex IV
fill(T[39], [
    ["Demonstrator video", "TODO", "End-to-end use case / relevant module behaviour"],
    ["Architecture diagram", "media/architecture_diagram.md", "System context and module location"],
    ["Sequence diagram or data-flow diagram", "media/architecture_diagram.md (Detect sequence)",
     "Inputs, outputs and interface logic"],
    ["Screenshots", "media/screenshots/ (add annotated frames)", "Visual behaviour, RViz, robot state"],
    ["Metrics or test results", "F1 96.88%, mAP 97.35%, 20/20 queries, 15 fps", "Accuracy / success rate"],
    ["End-user feedback", "KIRO D3 pilot (operator assistance)", "Industrial relevance"],
])

# =========================================================================== 3.4.3 Annex V
fill(T[40], [
    "The entire cv_tool module (node, Detect action, launch, config, bundled model, replay path).",
    "None for this module; KIRO's RaaS / module-licensing offering is separate.",
    "Yes — the recorded-rosbag replay path runs with no proprietary component.",
    "The live camera is replaced by a recorded RealSense rosbag for the hardware-free demo.",
    "KIRO RaaS / module licensing (IKNOWHOW SA).",
], right_only=True)

# =========================================================================== 3.4.4 Annex VI
fill(T[41], [
    ["Technical maintainer", "Nikos Kardaris, IKNOWHOW SA", "nick.kardaris@gmail.com / @nkardaris"],
    ["Project coordinator/contact person", "Maria Kampa; Angeliki Pilalitou, IKNOWHOW SA",
     "mkampa@iknowhow.com; apilalitou@iknowhow.com"],
    ["Commercial contact", "IKNOWHOW SA", "TODO"],
    ["ARISE mentor", "Politecnico di Milano (Mostafa Zarei)", "mostafa.zarei@polimi.it"],
])

# ---- ethics annex links + checklists ----
set_para_text("Final Ethics Assessment for this project is available here",
    "The document with justification, integration and/or future plans requested in the Final "
    "Ethics Assessment for this project is available here: see the KIRO Ethics & Human-Centricity "
    "Action Plan (D3 Annex II): "
    "https://docs.google.com/document/d/1YoPehbr1x2CY5RsyrH9cY20L4RYOOF_W/edit")
set_para_text("compiled Roadmap for future use for this project is available here",
    "The compiled Roadmap for future use for this project is available here: TODO (link once "
    "compiled).")

tick_checklists()

doc.save(DST)
print("Wrote", DST)
