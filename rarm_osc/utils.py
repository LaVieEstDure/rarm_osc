from ament_index_python.packages import get_package_share_directory
import xacro
import os


def get_urdf_from_xacro():
    package_path = get_package_share_directory('openarm_description')
    xacro_file_path = os.path.join(
        package_path, 'urdf', 'robot', 'v10.urdf.xacro')
    robot_config = xacro.process_file(str(xacro_file_path)).toxml()
    urdf_file_path = '/tmp/robotconfig.urdf'
    with open(urdf_file_path, 'w') as f:
        f.write(robot_config)
    return urdf_file_path 


# get_urdf_from_xacro()
