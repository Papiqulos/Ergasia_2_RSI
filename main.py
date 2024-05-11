import numpy as np # Linear Algebra
import pinocchio as pin # Pinocchio library
import os

from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

VISUALIZER = MeshcatVisualizer

def load_franka()->tuple[RobotWrapper, pin.Model, pin.Data, pin.GeometryModel, pin.GeometryData]:
    """
    Load the Franka Emika Panda robot model
    
    Returns:
    - robot: robot wrapper
    - model: pinocchio model
    - data: pinocchio data
    - geometry_model: pinocchio geometry model
    - geometry_data: pinocchio geometry data
    """
    # Load the URDF model
    current_path = os.path.abspath('') # where the folder `robot` is located at
    robot_path = os.path.join(current_path, "robot")

    # Read URDF model
    robot = RobotWrapper.BuildFromURDF(os.path.join(robot_path, "franka.urdf"), package_dirs = robot_path)

    # Extract pinocchio model and data
    model = robot.model
    data = robot.data

    geometry_model = pin.GeometryModel()
    geometry_data = pin.GeometryData(geometry_model)

    return robot, model, data, geometry_model, geometry_data

def step_world(current_state:np.ndarray, torques:np.ndarray, dt:float)->np.ndarray:
    """
    Step the world for a given time step dt
    
    Args:
    - torques: torques to be applied at each joint
    - dt: time step
    
    Returns:
    - states: list of states of the robot at each time step
    """
    robot, model, data, geometry_model, geometry_data = load_franka()

    aq0 = np.zeros(model.nv)
    
    b = pin.rnea(model, data, current_state, torques, aq0)

    M = pin.crba(model, data, current_state)

    aq = np.linalg.solve(M, torques - b)

    torques = torques + aq * dt

    new_state = pin.integrate(model, current_state, torques * dt)
    
    return new_state

def simulate(new_state:np.ndarray, torques:np.ndarray, T:int, dt:float)->np.ndarray:
    """
    Simulate the world for T seconds with a given time step dt
    
    Args:
    - model: pinocchio model
    - data: pinocchio data
    - new_state: new state of the robot
    - T: time to simulate
    - dt: time step
    
    Returns:
    - states: list of states of the robot at each time step
    """
    robot, model, data, geometry_model, geometry_data = load_franka()

    K = int(T/dt) + 1

    states = np.zeros((K, new_state.shape[0]))

    states[0] = new_state
    

    for k in range(1, K):
        states[k] = step_world(states[k-1], torques, dt)

        # check if the robot is in collision
        print(pin.computeCollisions(model, data, geometry_model, geometry_data, states[k], False))
            

    return states

def visualize(robot, state):

    # Visualize the robot
    robot.setVisualizer(VISUALIZER())
    robot.initViewer()
    robot.loadViewerModel("pinocchio")

def task1(): 

    model = load_franka()[1]
    T = 1
    dt = 0.1

    q0 = pin.randomConfiguration(model)
    torques = np.full_like(q0, 0.7)

    print(f"q0: {q0}")
    print(f"torques: {torques}")

    q = simulate(q0, torques, T, dt)

    end_state = q[-1]

    print(f"end_state : {end_state}")

if __name__ == "__main__":

    task1()