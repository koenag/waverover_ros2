from glob import glob
import os

from setuptools import setup

package_name = 'robot_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CSE276A',
    maintainer_email="chl235@ucsd.edu",
    description='Launch files for bringing up the robot',
    license='MIT',
)