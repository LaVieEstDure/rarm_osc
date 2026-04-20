import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation

from .controller import OperationalSpaceController

JOINT_NAMES = [
    'openarm_joint1',
    'openarm_joint2',
    'openarm_joint3',
    'openarm_joint4',
    'openarm_joint5',
    'openarm_joint6',
    'openarm_joint7',
]


class OSCNode(Node):
    def __init__(self):
        super().__init__('osc_controller')

        self._controller = OperationalSpaceController()
        n = len(JOINT_NAMES)

        self._q = np.zeros(n)
        self._qdot = np.zeros(n)
        self._name_to_idx: dict[str, int] | None = None

        # [des_pos(3), des_rot_flat(9), des_vel(3), des_omega(3)]
        self._z_ee_des = np.zeros(18)
        self._z_ee_des[3:12] = np.eye(3).flatten()  # identity rotation default

        self.create_subscription(JointState, 'joint_states', self._joint_state_cb, 10)
        self.create_subscription(PoseStamped, 'ee_target/pose', self._pose_cb, 10)
        self.create_subscription(TwistStamped, 'ee_target/twist', self._twist_cb, 10)

        self._cmd_pub = self.create_publisher(
            Float64MultiArray, 'forward_position_controller/commands', 10
        )

        self.create_timer(0.01, self._control_cb)  # 100 Hz

    def _joint_state_cb(self, msg: JointState):
        if self._name_to_idx is None:
            self._name_to_idx = {name: i for i, name in enumerate(msg.name)}

        for j, name in enumerate(JOINT_NAMES):
            idx = self._name_to_idx.get(name)
            if idx is None:
                continue
            if msg.position:
                self._q[j] = msg.position[idx]
            if msg.velocity:
                self._qdot[j] = msg.velocity[idx]

    def _pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        o = msg.pose.orientation
        self._z_ee_des[:3] = [p.x, p.y, p.z]
        rot = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        self._z_ee_des[3:12] = rot.flatten()

    def _twist_cb(self, msg: TwistStamped):
        v = msg.twist.linear
        w = msg.twist.angular
        self._z_ee_des[12:15] = [v.x, v.y, v.z]
        self._z_ee_des[15:18] = [w.x, w.y, w.z]

    def _control_cb(self):
        z = np.concatenate([self._q, self._qdot])
        tau = self._controller.control(z, self._z_ee_des)

        out = Float64MultiArray()
        out.data = tau.tolist()
        self._cmd_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OSCNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
