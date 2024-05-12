import numpy as np # Linear Algebra
import pinocchio as pin # Pinocchio library
import os
import time

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

def step_world(model:pin.Model, data:pin.Data, current_q:np.ndarray, current_u:np.ndarray, control_t:np.ndarray, dt:float)->tuple[np.ndarray, np.ndarray]:
    """
    Step the world for a given time step dt
    
    Args:
    - model: pinocchio model
    - data: pinocchio data
    - current_q: current position of each joint
    - current_u: current velocity of each joint
    - control_t: control torque to be applied at each joint
    - dt: time step
    
    Returns:
    - new_q: new position of each joint
    - current_t: new velocity of each joint
    """
    # Get joint limits
    joint_limit_up = model.upperPositionLimit
    joint_limit_low = model.lowerPositionLimit
    velocity_limit = model.velocityLimit

    # Integrate the dynamics and get the acceleration
    aq = pin.aba(model, data, current_q, current_u, control_t)

    # Integrate the acceleration to get the new velocity
    current_u += aq * dt

    # Integrate the velocity to get the new position
    new_q = pin.integrate(model, current_q, current_u * dt)

    # Position limit check
    if np.any(new_q > joint_limit_up) or np.any(new_q < joint_limit_low):
        print("Joint limit reached")
        new_q = current_q
        current_u = np.zeros(model.nv)

    # Velocity limit check
    if np.any(current_u > velocity_limit) or np.any(current_u < -velocity_limit):
        print("Velocity limit reached")
        current_u = np.zeros(model.nv)
    
    
    return new_q, current_u

def simulate(model:pin.Model, data:pin.Data, q:np.ndarray, control_t:np.ndarray, T:int, dt:float)->tuple[list, np.ndarray]:
    """
    Simulate the world for T seconds with a given time step dt
    
    Args:
    - model: pinocchio model
    - data: pinocchio data
    - control_t: torques to be applied at each joint
    - T: time to simulate
    - dt: time step
    
    Returns:
    - qs: list of joint params of the robot at each time step
    - end_state: final state of the robot
    """

    K = int(T/dt) 
    print(f"Time steps: {K}")

    u = np.zeros(model.nv)
    
    # start_state = np.array([q, u])

    qs = []
    # Simulate the world
    for _ in range(K):
        q, u = step_world(model, data, q, u, control_t, dt)
        qs.append(q)
            
    end_state = np.array([q, u])
    return qs, end_state

def visualize(robot:RobotWrapper, qs:list[np.ndarray]|np.ndarray):
    """
    Visualize the robot
    
    Args:
    - robot: robot wrapper
    - qs: list of states of the robot at each time step or a single state
    """

    # Visualize the robot
    robot.setVisualizer(VISUALIZER())
    robot.initViewer()
    robot.loadViewerModel("pinocchio")
    if isinstance(qs, list):
        for q in qs:
            robot.display(q)
            time.sleep(0.01)
    else:
        robot.display(qs)
    
def task1(): 

    robot, model, data, geometry_model, geometry_data = load_franka()

    T = 5
    dt = 0.01
    
    pin.seed(1083738)
    # q0 = pin.randomConfiguration(model)
    q0 = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])
    control_t = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    qs, end_state = simulate(model, data, q0, control_t, T, dt)
    print(f"End state: {end_state}")
    q, u = end_state
    visualize(robot, q)
    while True: continue
    

if __name__ == "__main__":

    task1()