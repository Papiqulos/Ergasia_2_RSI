import numpy as np
import Franka as fr
import matplotlib.pyplot as plt
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
        robot.visualize(qs)
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

    qs, end_state, errors = robot.simulate(q0, u, control_t, T, dt, target_q, Kp, Ki, Kd)
    q, u = end_state

    K = int(T/dt)
    print(f"Final error: {errors[-1]}")
    print(f"Final configuration: {q}")

    plt.plot(np.arange(K), errors)
    plt.xlabel("Time steps")
    plt.ylabel("Error")
    plt.title("Error vs Time steps")
    plt.grid()
    plt.show()


    
    # Visualize the robot
    if visuals:
        robot.visualize(qs)
        while True: continue
        
def task3(robot, T, dt, q0, u, control_t):    
    pass


if __name__ == "__main__":

    robot = fr.Franka()
    model = robot.model

    ## Simulation parameters
    # Time period
    T = 10
    # Time step
    dt = 0.01
    
    ## Initial state
    # Joints configuration
    pin.seed(10837383)
    q0 = pin.randomConfiguration(model)
    print(f"Initial configuration: {q0}")
    
    # Velocity
    u = np.zeros(model.nv)

    ## Initial Control torque
    control_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # task1(robot, T, dt, q0, u, control_t, visuals=True)

    Kp = 100.
    Ki = 0.
    Kd = 2.

    target_q = np.array([1.0, -0.645, -1.65, -2.15, -2.31,  2.18, 0.3])
    try:
        task2(robot, T, dt, q0, u, control_t, target_q, Kp, Ki, Kd)
    except KeyboardInterrupt:
        exit()

    # task3(robot, T, dt, q0, u, control_t)