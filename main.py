import numpy as np
import FrankaRobot as fr
import pinocchio as pin
  
def task1(robot:fr.Franka, T:int, dt:float, q0:np.ndarray, u:np.ndarray, control_t:np.ndarray, visuals:bool=False): 
    """
    Simulate the robot for a certain time period with a given control torque and visualize the robot
    
    Args:
    - robot: Franka robot object
    - T: time period
    - dt: time step
    - q0: initial position
    - u: initial velocity
    - control_t: control torque
    - visuals: whether to visualize the robot or not
    """

    # Simulate the robot
    qs, end_state = robot.simulate(q0, u, control_t, T, dt)
    q, u = end_state
    print(f"\t\t\tEnd state\nposistion: {end_state[0]}\nvelocity: {end_state[1]}")
    
    # Visualize the robot
    if visuals:
        robot.visualize(robot, qs)
        while True: continue

def task2(robot:fr.Franka, T:int, dt:float, q0:np.ndarray, u:np.ndarray, control_t:np.ndarray, target_q:np.ndarray, Kp:float=100., Ki:float=1., Kd:float=0., visuals:bool=False):
    """
    Simulate the robot for a certain time period with a given control torque 
    and use a Task Space Controller to reach a certain target pose profile 
    and visualize the robot
    
    Args:
    - robot: Franka robot object
    - T: time period
    - dt: time step
    - q0: initial position
    - u: initial velocity
    - control_t: control torque
    - target_q: target position
    - Kp: proportional gain
    - Ki: integral gain
    - Kd: derivative gain
    - visuals: whether to visualize the robot or not"""

    qs, end_state = robot.simulate(q0, u, control_t, T, dt, target_q, Kp, Ki, Kd)
    q, u = end_state

    print(f"\nEnd posistion:\t\tTarget position:")
    for i in range(len(q)):
        print(f"{q[i]:.3f}\t\t\t{target_q[i]:.3f}")
    
    # Visualize the robot
    if visuals:
        robot.visualize(robot, qs)
        while True: continue

def task3(robot, T, dt, q0, u, control_t):    
    pass


if __name__ == "__main__":

    robot = fr.Franka()
    model = robot.model

    # Time period
    T = 10

    # Time step
    dt = 0.1
    
    ## Initial state
    # Joints configuration
    q0 = pin.randomConfiguration(model)

    # Velocity
    u = np.zeros(model.nv)

    ## Control torque
    control_t = np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # task1(robot, T, dt, q0, u, control_t)

    Kp = 100.
    Ki = 1.
    Kd = 0.

    target_q = np.array([1.13380891, -0.56585662, -2.65877219, -2.05434817, -2.81552492,  0.18592432, 0.28310915])

    task2(robot, T, dt, q0, u, control_t, target_q, Kp, Ki, Kd)

    # task3(robot, T, dt, q0, u, control_t)