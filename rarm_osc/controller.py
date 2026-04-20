import jax
import frax
import numpy as np

from .utils import get_urdf_from_xacro

jax.config.update("jax_enable_x64", True)


class OperationalSpaceController:
    def __init__(self):
        robot = frax.Robot(get_urdf_from_xacro())
        q = np.zeros(robot.num_joints)
        print(robot.mass_matrix(q))

    def control(z, z_ee_des):
        return 0

if __name__ == "__main__":
    OperationalSpaceController()
