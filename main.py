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

def step_world(model:pin.Model, data:pin.Data, current_q:np.ndarray, current_t:np.ndarray, control_t:np.ndarray, dt:float)->tuple[np.ndarray, np.ndarray]:
    """
    Step the world for a given time step dt
    
    Args:
    - model: pinocchio model
    - data: pinocchio data
    - current_q: current state of the robot
    - current_t: torques to be applied at each joint
    - control_t: torques to be applied at each joint
    - dt: time step
    
    Returns:
    - new_q: new state of the robot
    - current_t: new torques to be applied at each joint
    """
    aq = pin.aba(model, data, current_q, current_t, control_t)

    current_t += aq * dt

    new_q = pin.integrate(model, current_q, current_t * dt)
    
    return new_q, current_t

def simulate(robot, model:pin.Model, data:pin.Data, control_t:np.ndarray, T:int, dt:float)->tuple[np.ndarray, np.ndarray]:
    """
    Simulate the world for T seconds with a given time step dt
    
    Args:
    - model: pinocchio model
    - data: pinocchio data
    - control_t: torques to be applied at each joint
    - T: time to simulate
    - dt: time step
    
    Returns:
    - states: list of states of the robot at each time step
    """

    K = int(T/dt) 
    print(f"Time steps: {K}")

    pin.seed(1083738)
    q = pin.randomConfiguration(model)
    u = np.zeros(model.nv)

    qs = []
    # Simulate the world
    for k in range(K):
        q, u = step_world(model, data, q, u, control_t, dt)
        qs.append(q)
            

    return q, u, qs

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

    T = 4
    dt = 0.01
    
    control_t = np.full(model.nv, 0.7)
    q, u, qs = simulate(robot, model, data, control_t, T, dt)

    print(f"end_q : {q}")
    visualize(robot, qs)
    while True: continue

if __name__ == "__main__":

    task1()