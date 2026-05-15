from setuptools import find_packages, setup
from glob import glob
from pathlib import Path
import os

package_name = 'cv_tool'

model_files = []
for path in Path('cv_tool/models').rglob('*'):
    if path.is_file():
        model_files.append(path)

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
]

for path in model_files:
    rel_dir = path.parent.relative_to('cv_tool')
    install_dir = os.path.join('share', package_name, str(rel_dir))
    data_files.append((install_dir, [str(path)]))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cv_tool = cv_tool.cv_tool:main'
        ],
    },
)
