from setuptools import setup, find_packages

setup(
    name="serl_robot_infra",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "pyrealsense2",
        "pymodbus==2.5.3",
        "opencv-python",
        "pyquaternion",
        "pyspacemouse",
        "hidapi",
        "pyyaml",
        "rospkg",
        "scipy",
        "requests",
        "flask",
        "defusedxml",
        "pygame>=2.0.0",
        "numpy>=1.24.3",  # 移除 <2.0 限制，允许使用 numpy 2.0（训练机器上使用 JAX 0.9.0 需要）
    ],
)
