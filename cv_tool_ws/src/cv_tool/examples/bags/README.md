# Demo rosbag (external download)

The **demo** ([`docs/04`](../../../../../docs/04_basic_demo_how_to_use.md)) replays a **recorded Intel
RealSense RGB-D rosbag** captured during the KIRO pilot at IKH. (The hello world does **not** need
this bag — it just starts the server; see [`docs/03`](../../../../../docs/03_installation_and_hello_world.md).)
Because raw RGB-D recordings are large (~1.6 GB for ~71 s), the bag is **not stored in this git
repository**. It is distributed as an external download and listed as an *external dependency* of the
demo.

## Download

| Item | Value |
|---|---|
| File | `boxes_0.db3` (rosbag2, sqlite3 storage) |
| Download link | **TODO: add public link (GitHub Release asset / Drive / Zenodo)** |
| Size | ~1.6 GB |
| Duration | ~71 s @ 15 Hz |
| Topics | `/camera/camera/color/image_raw` (`sensor_msgs/Image`, `bgr8`) |
|        | `/camera/camera/depth/image_rect_raw` (`sensor_msgs/Image`, `16UC1`, mm) |

These topic names match the defaults in [`config/config.yaml`](../../config/config.yaml),
so no remapping is required.

## Place / mount the bag

Mount the folder that contains the bag into the running Docker container (recommended, keeps
it out of the image):

```bash
docker run --rm -it --net=host \
    -v /absolute/path/to/rosbags:/cv_tool_ws/bags \
    cv_tool:humble \
    bash
```

If the download does **not** include a `metadata.yaml` next to the `.db3`, regenerate it once:

```bash
ros2 bag reindex /cv_tool_ws/bags -s sqlite3
```

## Run

```bash
# Terminal 1 (inside the container): replay the bag + start the action server
ros2 launch cv_tool cv_tool_demo.launch.py bag_path:=/cv_tool_ws/bags/boxes_0.db3

# Terminal 2 (inside the container): request a tool
source /cv_tool_ws/install/setup.bash
ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect \
    "{tool_name: screwdriver}" --feedback
```

See [`docs/04_basic_demo_how_to_use.md`](../../../../../docs/04_basic_demo_how_to_use.md) for the full
demo walkthrough and expected output.
