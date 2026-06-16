from setuptools import find_packages, setup
from glob import glob
from pathlib import Path
import os

package_name = 'cv_tool'

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
]


def install_tree(source_root, dest_root):
    """Install every file under source_root into dest_root, preserving structure."""
    for path in Path(source_root).rglob('*'):
        if path.is_file():
            rel_dir = path.parent.relative_to(source_root)
            install_dir = os.path.normpath(
                os.path.join('share', package_name, dest_root, str(rel_dir)))
            data_files.append((install_dir, [str(path)]))


# Ship the bundled YOLO/OpenVINO model(s) and the example assets (recorded rosbag,
# sample payloads, hello-world helpers) inside the package share.
install_tree('cv_tool/models', 'models')
if Path('examples').is_dir():
    install_tree('examples', 'examples')

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='Nikos Kardaris',
    maintainer_email='nick.kardaris@gmail.com',
    description=(
        'ARISE-KIRO tool recognition module: a ROS 2 / Vulcanexus action server that detects '
        'industrial tools for robotic picking using Ultralytics YOLO (OpenVINO) and deprojects '
        '2D bounding boxes into 3D metric coordinates using aligned RGB-D camera data.'
    ),
    license='AGPL-3.0-or-later',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cv_tool = cv_tool.cv_tool:main',
        ],
    },
)
