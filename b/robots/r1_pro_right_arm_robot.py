import os
import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel


class R1ProRightArm(ManipulatorModel):
    """
    R1 Pro right arm (7-DOF).
    """

    def __init__(self, idn=0):
        xml_path = os.path.join(os.path.dirname(__file__), "r1_pro_right_arm", "robot.xml")
        super().__init__(xml_path, idn=idn)

    @property
    def default_mount(self):
        return "PhantomMount"

    @property
    def default_gripper(self):
        return "R1ProGripper"

    @property
    def default_controller_config(self):
        return "default_r1_pro_right_arm"

    @property
    def init_qpos(self):
        return np.array([0.0, -1.5, 0.0, -1.0, 0.0, 0.5, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
