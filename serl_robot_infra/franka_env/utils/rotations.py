from scipy.spatial.transform import Rotation as R
import numpy as np
from pyquaternion import Quaternion


def quat_2_euler(quat):
    """calculates and returns: yaw, pitch, roll from given quaternion"""
    return R.from_quat(quat).as_euler("xyz")


def euler_2_quat(xyz):
    """
    Converts euler angles (roll, pitch, yaw) in 'xyz' intrinsic order to quaternion [x, y, z, w].
    Note: Updated to return XYZW to match Scipy/SERL standards.
    """
    from scipy.spatial.transform import Rotation as R
    return R.from_euler("xyz", xyz).as_quat()
