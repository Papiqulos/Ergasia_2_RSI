import numpy as np # Linear Algebra
import pinocchio as pin # Pinocchio library
import os
import time
from control import pid_torque_control, fk_all

from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

VISUALIZER = MeshcatVisualizer

class Franka:
    """Franka Panda robot class"""
    
    def __init__(self):
        self.robot, self.model, self.data = self.load_franka()

        # Initialize the visualizer
        self.robot.setVisualizer(VISUALIZER())
        self.robot.initViewer()
        self.robot.loadViewerModel("pinocchio")

    def step_world(self, current_q:np.ndarray, current_u:np.ndarray, control_t:np.ndarray, dt:float)->tuple[np.ndarray, np.ndarray]:
        """
        Step the world for a given time step dt
        
        Args:
        - model: pinocchio model
        - data: pinocchio data
        - current_q: current position of each joint (angles of each joint)
        - current_u: current velocity of each joint
        - control_t: control torque to be applied at each joint
        - dt: time step
        
        Returns:
        - new_q: new position of each joint
        - current_t: new velocity of each joint
        """
        # Get joint limits
        joint_limit_up = self.model.upperPositionLimit
        joint_limit_low = self.model.lowerPositionLimit
        velocity_limit = self.model.velocityLimit

        # Integrate the dynamics and get the acceleration
        aq = pin.aba(self.model, self.data, current_q, current_u, control_t)

        # Integrate the acceleration to get the new velocity
        current_u += aq * dt

        # Integrate the velocity to get the new position
        new_q = pin.integrate(self.model, current_q, current_u * dt)

        # Clip the position if it exceeds the joint limits
        new_q = np.clip(new_q, joint_limit_low, joint_limit_up)

        # Clip the velocity if it exceeds the velocity limits
        current_u = np.clip(current_u, -velocity_limit, velocity_limit)
        
        return new_q, current_u

    def load_franka(self)->tuple[RobotWrapper, pin.Model, pin.Data, pin.GeometryModel, pin.GeometryData]:
        """
        Load the Franka Panda Panda robot model
        
        Returns:
        - robot: robot wrapper
        - model: pinocchio model
        - data: pinocchio data
        """
        # Load the URDF model
        current_path = os.path.abspath('') # where the folder `robot` is located at
        robot_path = os.path.join(current_path, "robot")

        # Read URDF model
        robot = RobotWrapper.BuildFromURDF(os.path.join(robot_path, "franka.urdf"), package_dirs = robot_path)

        # Extract pinocchio model and data
        model = robot.model
        data = robot.data

        return robot, model, data

    def simulate(self, q:np.ndarray, u:np.ndarray, control_t:np.ndarray, T:int, dt:float, target_q:np.ndarray|None=None, Kp:float = 120., Ki:float = 0., Kd:float = 0.1)->tuple[np.ndarray, np.ndarray]:
        """
        Simulate the world for T seconds with a given time step dt with or without a PID torque controller
        
        Args:
        - model: pinocchio model
        - data: pinocchio data
        - q: initial position of each joint
        - u: initial velocity of each joint
        - control_t: torques to be applied at each joint
        - T: time to simulate
        - dt: time step
        - target_q: target position of each joint
        - Kp: proportional gain
        - Ki: integral gain
        - Kd: derivative gain
        
        Returns:
        - qs: list of joint params of the robot at each time step
        - end_state: final state of the robot (position and velocity)
        - errors: list of the norm of the error at each time step
        """

        # Number of time steps
        K = int(T/dt) 
        print(f"Time steps: {K}")

        # Store the positions of the robot at each time step
        qs = np.zeros((K, self.model.nq))

        # Initialize the error
        error = None
        errors = np.zeros(K)

        # Simulate the world
        if target_q is not None:

            # Target pose profile
            target_T = self.get_pose_profile(target_q)

            for i in range(K):

                # PID torque control
                control_t, error= pid_torque_control(self.model, self.data, target_T, q, dt, Kp, Ki, Kd)
                errors[i] = np.linalg.norm(error)
                q, u = self.step_world(q, u, control_t, dt)
                qs[i] = q
        else:
            for i in range(K):

                # No control
                q, u = self.step_world(q, u, control_t, dt)
                qs[i] = q
        
        # End state
        end_state = np.array([q, u])

        return qs, end_state, errors
    
    def get_pose_profile(self, target_q:np.ndarray)->pin.SE3:
        """
        Get the pose profile of the robot 
        
        Args:
        - model: pinocchio model
        - data: pinocchio data
        - target_q: target position of each joint
        
        Returns:
        - target_T: target pose profile
        """
        # End effector frame id
        frame_id = self.model.getFrameId("panda_ee")

        # Forward kinematics
        fk_all(self.model, self.data, target_q)

        # Get the transformation matrix
        target_T = self.data.oMf[frame_id].copy()
        
        return target_T

    def visualize(self, qs:np.ndarray):
        """
        Visualize the robot
        
        Args:
        - robot: robot wrapper
        - qs: list of states of the robot at each time step or a single state
        """

        # Visualize the robot
        if len(qs) == self.model.nq:
            self.robot.display(qs)
        else:
            for q in qs:
                self.robot.display(q)
                time.sleep(0.01)