import numpy as np # Linear Algebra
import pinocchio as pin # Pinocchio library
import os

from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

VISUALIZER = MeshcatVisualizer

def load_franka()->tuple[RobotWrapper, pin.Model, pin.Data]:
    """
    Load the Franka Emika Panda robot model
    
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

def step_world(model:pin.Model, current_state:np.ndarray, torques:np.ndarray, T:int, dt:float)->np.ndarray:
    """
    Simulate the world for T seconds with a given time step dt
    
    Args:
    - model: pinocchio model
    - current_state: current state of the robot
    - torques: torques to be applied at each joint
    - T: time to simulate
    - dt: time step
    
    Returns:
    - states: list of states of the robot at each time step"""

    new_state = 0
    K = int(T/dt) + 1

    new_state = pin.integrate(model, current_state, torques * dt)
    
    return new_state

def visualize(robot, state):

    # Visualize the robot
    robot.setVisualizer(VISUALIZER())
    robot.initViewer()
    robot.loadViewerModel("pinocchio")

def task1(): 
    robot, model, data = load_franka()

    state = pin.neutral(model)
    torques = np.random.rand(model.nv) * 10

    print(f"Initial state:\t{state}")
    print(f"Torques:\t{torques}")

    new_state = step_world(model, state, torques, 4, 0.01)

    print(f"New state:\t{new_state}")

if __name__ == "__main__":

    task1()