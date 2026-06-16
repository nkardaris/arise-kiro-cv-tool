# examples/

Assets for running `cv_tool` without the original industrial hardware.

| Path | Purpose |
|---|---|
| `bags/README.md` | How to download and replay the recorded RealSense RGB-D demo rosbag (the canonical hardware-free hello world / basic demo input). |

The recorded rosbag itself is an **external download** (too large for git) — see
[`bags/README.md`](bags/README.md). Run it with:

```bash
ros2 launch cv_tool cv_tool_replay.launch.py bag_path:=/cv_tool_ws/bags/boxes_0.db3
```
