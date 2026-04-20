FROM eprosima/vulcanexus:humble-desktop

WORKDIR /cv_tool_ws/

RUN apt-get update && apt-get install -y \
    python3-pip python3-opencv ffmpeg nano vim
RUN pip3 install --no-cache-dir torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir ultralytics "numpy<2.0.0" "lap>=0.5.12" debugpy "openvino>=2024.0.0"

COPY cv_tool_ws/src/ ./src/

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

CMD ["/bin/bash", "-lc", "source /cv_tool_ws/install/setup.bash && exec ros2 launch cv_tool cv_tool.launch.py"]
