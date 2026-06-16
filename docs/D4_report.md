# D4 — Shareable HRI Module: written report (DRAFT)

> **Draft** of the ARISE D4 written report for the KIRO **Tool recognition** module. Transfer this
> content into the official ARISE D4 Word/PDF template (Exo 2, size 10) before submission. Items
> marked **TODO** require team input. Repository content is referenced via links rather than copied.

---

## 3.1 Project identification sheet

| Field | Information |
|---|---|
| Project title | Key Intelligent & Interactive Robotic Operator |
| Project acronym | KIRO |
| Lead organisation | IKNOWHOW SA (IKH) |
| Contact person | Maria Kampa, mkampa@iknowhow.com, IKH; Angeliki Pilalitou, apilalitou@iknowhow.com, IKH |
| Technical contact for the repository | Nikos Kardaris, nick.kardaris@gmail.com, GitHub @nkardaris |
| Assigned ARISE mentor | Politecnico di Milano — Mostafa Zarei (mostafa.zarei@polimi.it), Amirshayan Nasirimajd, Nima Rahmani, Walter Quadrini |
| Reusable module name | `cv_tool` (Tool recognition) |
| GitHub repository URL | https://github.com/nkardaris/arise-kiro-cv-tool |
| Release/tag submitted for review | **TODO: v0.1.0 (tag after merge)** |
| Demonstrator video URL | **TODO** |

## 3.2 GitHub repository (summary)

The repository is the technical source of truth. Identification:

| Item | Value |
|---|---|
| Repository name | arise-kiro-cv-tool |
| Owner | nkardaris |
| URL | https://github.com/nkardaris/arise-kiro-cv-tool |
| Visibility | **TODO: Public** |
| Submitted release/tag | **TODO: v0.1.0** |
| Main branch | main |
| Primary language | Python |
| Primary runtime | ROS 2 / Vulcanexus Humble (Docker) |
| Issue tracker | Yes (GitHub Issues) |

**License & ownership:** GNU **AGPL-3.0-or-later** ([LICENSE](../LICENSE)); copyright IKNOWHOW SA /
KIRO. Third-party: Ultralytics YOLO (AGPL-3.0), OpenVINO (Apache-2.0), PyTorch (BSD), OpenCV
(Apache-2.0), ROS 2/Vulcanexus packages. Maintainer: Nikos Kardaris (@nkardaris). Maintenance: best
effort through project end. **External dependency:** the demo rosbag (`boxes_0.db3`) is downloaded
separately — see [`examples/bags/README.md`](../cv_tool_ws/src/cv_tool/examples/bags/README.md).

**Scope of the open implementation:** the full detection capability is open — node, `Detect` action,
launch files, config, bundled YOLO/OpenVINO model, and a documented hardware-free replay path. The
grasping/AprilTag logic, mission controller and robot/camera drivers remain demonstrator-specific
and are out of scope (see §3.3.8).

Repository structure, README content, interface tables and the install/hello-world/demo material are
documented in the repo: [README](../README.md), [`docs/02_interfaces.md`](02_interfaces.md),
[`docs/03_installation_and_hello_world.md`](03_installation_and_hello_world.md),
[`docs/04_basic_demo_how_to_use.md`](04_basic_demo_how_to_use.md).

---

## 3.3 The written report

### 3.3.1 Executive module summary

| Question | Response |
|---|---|
| What is the reusable module? | A ROS 2 / Vulcanexus action server that detects a requested industrial tool in an RGB-D stream (Ultralytics YOLO on OpenVINO), deprojects the 2D bounding box into 3D metric coordinates using depth + camera intrinsics, and returns the tool's 3D center/top-left/bottom-right points and confidence. It accepts a detection only when the tool is centered and stable across consecutive frames, and disambiguates visually identical tools of different size from their measured dimensions. |
| What problem does it solve? | During robotic tool delivery, finding and precisely localizing a specific requested tool among many in trays/boxes, including size-ambiguous tools, so a manipulator can grasp it reliably. |
| What does it provide off-the-shelf? | Real-time (≈15 fps CPU) tool detection + 3D localization via a single ROS 2 action; size disambiguation; centering-stability gating; a bundled model and a no-hardware rosbag demo. |
| Who is the intended user? | Robotics developers / integrators building pick-and-deliver or bin-picking pipelines on ROS 2/Vulcanexus; the Vulcanexus community; ARISE ecosystem adopters. |
| Minimum reproducible execution path | Docker (Vulcanexus Humble) + recorded RealSense rosbag replay — no camera or robot required. |
| Main evidence of validation | KIRO D3 pilot at IKH: F1 96.88 %, mAP 97.35 %, 20/20 tool queries detected; demonstrator video (TODO). |

### 3.3.2 Relation with ARISE and previous milestones

| Milestone | Relevant result | Reflected in the module |
|---|---|---|
| Stage 1 — IMP | Need: robot tool delivery; CV to identify/pick tools with high precision | Defined the tool-recognition scope |
| Stage 2 — PoC / D2.2 MVP | Tool recognition MVP; UR10e + RealSense integration | Detection + 3D localization approach retained |
| Stage 3 — TRL6-7 demonstrator (D3) | Validated tool recognition in the IKH pilot (F1 96.88 %, mAP 97.35 %, 20/20 queries) | This action server is the extracted, validated capability |
| Stage 4 — Shareable module (D4) | Open implementation + documentation + hardware-free demo | Now reusable by third parties |

**ARISE alignment narrative.** The module runs on Vulcanexus Humble and exposes its capability over
standard ROS 2 (one action over Fast DDS). FIWARE/NGSI-LD and the DDS↔NGSI-LD enabler are handled
centrally in KIRO over the Robot/Mission/Worker entities, not by this perception node; ROS4HRI is
applied by KIRO's human-facing modules. The applicability of each interface is justified in
[`docs/02_interfaces.md`](02_interfaces.md#arise-middleware-interfaces--applicability) — to be
confirmed with the ARISE mentor.

### 3.3.3 Platforms, missions and tasks

| Platform | Tested | Expected | Limitations |
|---|---|---|---|
| Manipulator/cobot | UR10e (mobile-base mounted) | Any arm (module is platform-agnostic) | Needs an RGB-D camera in the work area |
| Sensor | Intel RealSense RGB-D | Any aligned `bgr8`+`16UC1` RGB-D source | Mono/RGB-only loses 3D + size disambiguation |
| Simulator/recorded data | RealSense rosbag (sqlite3) | Equivalent RGB-D bag/sim | — |

Missions: **collaborative assembly / operator assistance — on-demand tool delivery** (Yes);
object handover, navigation, inspection, intralogistics (No — other KIRO modules / N/A).
Task: **tool identification + 3D localization for picking** — input RGB-D + tool id → output 3D
keypoints + confidence — status *implemented/tested*.

### 3.3.4 Off-the-shelf capabilities

| Capability | Input | Output | Interface | Status |
|---|---|---|---|---|
| Tool detection | RGB | class + 2D bbox + confidence | ROS 2 action | tested |
| 3D localization | RGB + depth + intrinsics | 3D center/top_left/bottom_right | ROS 2 action | tested |
| Size disambiguation | depth bbox patch | small vs large class | internal | tested |
| Centering-stability gating | bbox over N frames | accept stable, centered only | params | tested |

### 3.3.5 Interoperability evidence

| Area | Evidence | Link |
|---|---|---|
| ROS 2/Vulcanexus | Node, action, topics, QoS, params, launch files | [`docs/02_interfaces.md`](02_interfaces.md) |
| FIWARE/NGSI-LD | N/A justification + candidate `ToolDetection` mapping | [`docs/02_interfaces.md`](02_interfaces.md#fiware--ngsi-ld--na-handled-centrally-in-kiro) |
| DDS↔NGSI-LD enabler | N/A justification (central enabler at system level) | [`docs/02_interfaces.md`](02_interfaces.md#ddsngsi-ld-enabler--na-here) |
| ROS4HRI/ROS4RI | N/A justification (no human perception) | [`docs/02_interfaces.md`](02_interfaces.md#ros4hri--ros4ri--na-no-human-perception) |

### 3.3.6 Added value through Vulcanexus / ROS 2

| Question | Response |
|---|---|
| How does the module use Vulcanexus? | Vulcanexus Humble image (`eprosima/vulcanexus:humble-desktop`); ROS 2 Humble action server + image subscriptions; launch files; CPU OpenVINO inference. |
| Vulcanexus-specific features | Fast DDS transport for the action and image streams; `--net=host` discovery; rosbag2 replay for hardware-free reproduction. |
| Contribution to the ecosystem | A reusable, packaged ROS 2 HRI/perception capability (tool detection + 3D localization) with a documented Docker + rosbag demo; candidate for a Vulcanexus example/tutorial. |
| Could it join a Vulcanexus metapackage/example set? | Yes — it is self-contained; would need a publicly hosted demo bag + tag. |
| Evidence | Repository, this report, annotated frames, metrics from D3; demonstrator video (TODO). |
| What remains outside Vulcanexus | Pip-only ML deps (Ultralytics/OpenVINO/PyTorch); the robot/camera drivers and grasping logic. |

### 3.3.7 Installation, hello world & demo evidence

Docker build → run with the demo rosbag mounted → `cv_tool_replay.launch.py` → `Detect` action goal
returns `success: true` with 3D points. Full commands and expected output:
[`docs/03_installation_and_hello_world.md`](03_installation_and_hello_world.md) and
[`docs/04_basic_demo_how_to_use.md`](04_basic_demo_how_to_use.md). The hello world needs **no
hardware** (recorded RealSense bag). Troubleshooting and a simulation/mock path are documented.

### 3.3.8 Role in the TRL6-7 demonstrator

| Demonstrator info | Response |
|---|---|
| Demonstrator | KIRO D3 — Demonstration of HRI-enabled solution at work |
| Environment | IKH facilities, ARISTOS assembly area (real factory floor), pilot |
| Robot/platform | UR10e + mobile base, RealSense RGB-D, PowerPick 10 vacuum gripper |
| End user / scenario | Shop-floor operators, on-demand tool delivery |
| Problem | Find + localize a requested tool (incl. size-ambiguous ones) for picking |
| Module role | `tool_recognition` skill in `SearchToolState`/`pick_up_tool`; triggers "stop & grasp" and supplies 3D keypoints |
| Video | **TODO** |

Extraction vs demonstrator-specific parts: see
[`docs/05_role_in_demonstrator.md`](05_role_in_demonstrator.md).

### 3.3.9 Validation, impact & exploitation

| Evidence | Result | Relevance |
|---|---|---|
| Metrics (validation) | 98.01 % P, 98.46 % R, 99.23 % mAP, ~30 ms/frame | Detection quality |
| Metrics (pilot) | F1 96.88 %, mAP 97.35 %, 20/20 queries, 15 fps | Real-environment performance (> 95 % KPI) |
| Demonstrator video | **TODO** | End-to-end behaviour |
| Screenshots | [`media/screenshots/`](../media/screenshots/) (**add frames**) | Visual behaviour |
| Impact (end user) | Reduced tool-fetching overhead, fewer interruptions; supports operator productivity/ergonomics | Industrial relevance |
| Impact (tech provider) | Reusable ROS 2 detection+localization block for pick/bin-picking pipelines | Reuse potential |

### 3.3.10 Openness, commercial boundary & limitations

| Topic | Response |
|---|---|
| Open boundary | Node, `Detect` action, launch, config, bundled model, rosbag replay path — all open (AGPL-3.0). |
| Commercial/proprietary | None withheld for this module; KIRO's RaaS/integration offering is separate. |
| Technical limitations | Depth-dependent size disambiguation (fixed ~0.17 m threshold; needs reliable depth 0.2–1.0 m); tuned to the KIRO tool set/tray layout; CPU 15 fps. |
| Hardware limitations | Needs an aligned RGB-D source with known intrinsics. |
| Untested cases | Other tool sets/backgrounds/distances; non-RealSense cameras; GPU path. |
| Future work | Broaden tool set/retraining; optional NGSI-LD publication; optional ROS4HRI pairing for tool-to-operator presentation. |
| Ethics/safety/privacy | Perceives tools, not people; no personal data. KIRO ethics handled at system level (see Annex). |

### 3.3.11 Self-assessment for ARISE ecosystem visibility

| Level | Selection | Justification |
|---|---|---|
| Related | Yes | Meets all minimum requirements; can be listed with a link. |
| **Featured** | **Yes (target)** | Good reproducibility (Docker + rosbag), clear README + docs, strong validated metrics, clean ROS 2/Vulcanexus interface. |
| Flagship | No | FIWARE/ROS4HRI are (justifiably) N/A for this perception module, and a public demo video/bag link are pending. |

| Area | Evidence |
|---|---|
| Reusability | Platform-agnostic action interface; only consumes camera topics. |
| Reproducibility | Docker image + hardware-free rosbag hello world with expected output. |
| ARISE interoperability | ROS 2/Vulcanexus core; FIWARE/DDS/ROS4HRI justified N/A. |
| Contribution to Vulcanexus | Packaged reusable HRI perception capability + Docker/rosbag demo. |
| Validation | D3 pilot metrics + (TODO) demonstrator video. |
| Documentation quality | README + 5 docs pages + interface tables + Mermaid diagrams. |
| Sustainability | AGPL-3.0, named maintainer, issue tracker, planned v0.1.0 tag. |

### 3.3.12 Written report final checklist

- [x] Project identification table completed (video/tag TODO)
- [x] Repository URL provided; release/tag **TODO**
- [x] Open implementation scope described
- [x] README and repository structure described
- [x] ROS 2/Vulcanexus, FIWARE/NGSI-LD, DDS enabler, ROS4HRI evidence summarised (N/A justified)
- [x] Installation, hello world and basic demo evidence included
- [x] TRL6-7 demonstrator role explained
- [ ] Video and visual evidence linked (**TODO**)
- [x] Limitations, proprietary boundaries and future work stated
- [x] Annexes and previous deliverables linked

---

## 3.4 Annexes

### Annex I — Previous milestones
IMP, D2.1 (+video), D2.2: links in [`media/video_link.md`](../media/video_link.md) (from D3 Annex I).
**D3 evaluation recommendations:** *to be copied here once the D3 evaluation letter is received,
with replies.* (TODO)

### Annex IV — Demonstrator evidence
Demonstrator video (**TODO**); architecture/sequence diagrams in
[`media/architecture_diagram.md`](../media/architecture_diagram.md); screenshots in
[`media/screenshots/`](../media/screenshots/) (**add frames**); metrics in §3.3.9.

### Annex V — Commercial / proprietary clarification
Open: the entire `cv_tool` module. Proprietary: none for this module. The module runs without any
proprietary element (rosbag replay path). No wrapper/mock substitution was needed beyond the
recorded-data demo path.

### Annex VI — Contacts
| Role | Name / org | Contact |
|---|---|---|
| Technical maintainer | Nikos Kardaris | nick.kardaris@gmail.com / @nkardaris |
| Project contact | Maria Kampa, Angeliki Pilalitou, IKNOWHOW SA | mkampa@iknowhow.com, apilalitou@iknowhow.com |
| ARISE mentor | Politecnico di Milano | mostafa.zarei@polimi.it |

### Annex — Ethics & Roadmap for Future Use
KIRO Ethics & Human-Centricity Action Plan: handled at the KIRO system level (D3 Annex II).
Roadmap for Future Use: **TODO (link once compiled)**.
