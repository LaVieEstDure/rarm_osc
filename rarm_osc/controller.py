import jax
import jax.numpy as jnp
import frax
import numpy as np
from frax.utils.rotation_utils import orientation_error_3D

from .utils import get_urdf_from_xacro

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")


class OperationalSpaceController:
    def __init__(
        self,
        kp_pos=50.0,
        kp_rot=20.0,
        kd_pos=20.0,
        kd_rot=10.0,
        kp_joint=10.0,
        kd_joint=5.0,
        des_q=None,
    ):
        robot = frax.Robot(get_urdf_from_xacro())
        n = robot.num_joints

        kp_task = jnp.array(np.concatenate(
            [kp_pos * np.ones(3), kp_rot * np.ones(3)]))
        kd_task = jnp.array(np.concatenate(
            [kd_pos * np.ones(3), kd_rot * np.ones(3)]))
        kp_j = jnp.array(kp_joint * np.ones(n))
        kd_j = jnp.array(kd_joint * np.ones(n))
        _des_q = jnp.zeros(n) if des_q is None else jnp.asarray(des_q)
        _des_qdot = jnp.zeros(n)
        _des_accel = jnp.zeros(6)  # [linear; angular] feed-forward
        _joint_limits = jnp.asarray(robot.joint_max_forces)

        @jax.jit
        def _compute(z, z_ee_des):
            q, qdot = z[:n], z[n:2 * n]
            des_pos = z_ee_des[:3]
            des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
            des_vel = z_ee_des[12:15]
            des_omega = z_ee_des[15:18]

            M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
            pos, rot = ee_tmat[:3, 3], ee_tmat[:3, :3]
            twist = J @ qdot
            vel, omega = twist[:3], twist[3:]

            pos_error = pos - des_pos
            vel_error = vel - des_vel
            rot_error = orientation_error_3D(rot, des_rot)
            omega_error = omega - des_omega
            task_p_error = jnp.concatenate([pos_error, rot_error])
            task_d_error = jnp.concatenate([vel_error, omega_error])

            task_inertia_inv = J @ M_inv @ J.T
            task_inertia = jnp.linalg.inv(task_inertia_inv)
            J_bar = M_inv @ J.T @ task_inertia

            task_accel = _des_accel - kp_task * task_p_error - kd_task * task_d_error
            tau = J.T @ (task_inertia @ task_accel) + g + c

            # Null space task
            NT = jnp.eye(n) - J.T @ J_bar.T
            joint_accel = -kp_j * (q - _des_q) - kd_j * (qdot - _des_qdot)
            tau += NT @ (M @ joint_accel)

            return jnp.clip(tau, -_joint_limits, _joint_limits)

        self.robot = robot
        self._compute = _compute

    def control(self, z, z_ee_des):
        return self._compute(z, z_ee_des)


if __name__ == "__main__":
    OperationalSpaceController()
