import numpy as np
import Franka as fr
import matplotlib.pyplot as plt
import pinocchio as pin
from control import fk_all

init = False
qp_init = False
prev_error = None
sum_error = 0.

init_null = False
sum_null = 0.
prev_null = None

def pid_torque_control(model:pin.Model, data:pin.Data, target_T:np.ndarray, current_q:np.ndarray, dt:float, Kp:float = 120., Ki:float = 0., Kd:float = 0.1, Kp_theta:float = 10., Ki_theta:float = 0., Kd_theta:float = 0.)->tuple[np.ndarray, np.ndarray]:
    """
    PID torque controller with null space controller

    Args:
    - model : Pinocchio model
    - data : Pinocchio data
    - target_T : target pose profile
    - current_q : current joint configuration
    - dt : time step
    - Kp : proportional gain
    - Ki : integral gain
    - Kd : derivative gain
    - Kp_theta : proportional gain for null space controller
    - Ki_theta : integral gain for null space controller
    - Kd_theta : derivative gain for null space controller
    
    Returns:
    - control_t : control torque
    - error : error
    """
    global init
    global prev_error
    global sum_error
    global init_null
    global sum_null
    global prev_null

    frame_id = model.getFrameId("panda_ee")

    # Compute current transformation matrix
    fk_all(model, data, current_q)
    current_T = data.oMf[frame_id].copy()

    current_rotation = current_T.rotation
    current_translation = current_T.translation

    target_rotation = target_T.rotation
    target_translation = target_T.translation   

    # Compute error
    error_rotation = pin.log(target_rotation @ current_rotation.T)
    error_translation = target_translation - current_translation

    error = np.concatenate((error_translation, error_rotation))
    # print(f"Error: {error}")

    if not init:
        prev_error = np.copy(error)

    
    sum_error += (error * dt)
    diff_error = (error - prev_error) / dt

    error = Kp * error + Kd * diff_error + Ki * sum_error

    prev_error = np.copy(error)

    if not init:
        sum_error = 0.
        prev_error = None
        init = False
    else:
        init = True

    # Compute Jacobian
    J = pin.computeFrameJacobian(model, data, current_q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED) # Jacobian in world frame

    # Compute control torque
    # Without null space controller
    control_t = J.T @ error

    # With null space controller
    # One implementation of the null space controller
    q_target = (model.upperPositionLimit - model.lowerPositionLimit) / 2. + model.lowerPositionLimit
    error_null = current_q - q_target

    if not init_null:
        prev_null = np.copy(error_null)

    sum_null += error_null * dt
    diff_null = (error_null - prev_null) / dt
    t_reg = Kp_theta * error_null - Ki_theta * sum_null -Kd_theta * diff_null
    prev_null = np.copy(error_null)

    if not init_null:
        sum_null = 0.
        prev_null = None
        init_null = False
    else:
        init_null = True

    control_t += (np.eye(model.nv) - J.T @ np.linalg.pinv(J.T)) @ t_reg

    return control_t, error
  
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

def task2(robot:fr.Franka, T:int, dt:float, q0:np.ndarray, u:np.ndarray, control_t:np.ndarray, target_q:np.ndarray, Kp:float=100., Ki:float=1., Kd:float=0., Kp_theta:float = 10., Ki_theta:float = 0., Kd_theta:float = 0.1, visuals:bool=False):
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
    - Kp_theta: proportional gain for null space controller
    - Ki_theta: integral gain for null space controller
    - Kd_theta: derivative gain for null space controller
    - visuals: whether to visualize the robot or not
    """

    qs, end_state, errors = robot.simulate(q0, u, control_t, T, dt, target_q, Kp, Ki, Kd)
    q, u = end_state

    K = int(T/dt)
    print(f"Final error norm: {errors[-1]}")
    # print("-----------------------------------")
    print(f"Initial configuration: {q0}")
    print(f"Final configuration: {q}")
    print(f"Target configuration: {target_q}")

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
        
def task3(robot:fr.Franka, T:int, dt:float, q0:np.ndarray, u:np.ndarray, control_t:np.ndarray, target_q1:np.ndarray, target_q2:np.ndarray, target_q3:np.ndarray, target_q4:np.ndarray, Kp:float=100., Ki:float=1., Kd:float=0., Kp_theta:float = 10., Ki_theta:float = 0., Kd_theta:float = 0., visuals:bool=False):    
    """ 
    Testing the null space controller in four different pose profiles

    Args:
    - robot: Franka robot object
    - T: time period
    - dt: time step
    - q0: initial position
    - u: initial velocity
    - control_t: control torque
    - target_q1: target position 1
    - target_q2: target position 2
    - target_q3: target position 3
    - target_q4: target position 4
    - Kp: proportional gain
    - Ki: integral gain
    - Kd: derivative gain
    - Kp_theta: proportional gain for null space controller
    - Ki_theta: integral gain for null space controller
    - Kd_theta: derivative gain for null space controller
    - visuals: whether to visualize the robot or not 
    """

    # Simulate the robot for each target pose profile
    print("-----------------------------------")
    target_T1 = robot.get_pose_profile(target_q1)
    print(f"Target pose profile 1: {target_T1}")
    task2(robot, T, dt, q0, u, control_t, target_q1, Kp, Ki, Kd, Kp_theta, Ki_theta, Kd_theta, visuals)
    

    print("-----------------------------------")
    target_T2 = robot.get_pose_profile(target_q2)
    print(f"Target pose profile 2: {target_T2}")
    task2(robot, T, dt, q0, u, control_t, target_q2, Kp, Ki, Kd, Kp_theta, Ki_theta, Kd_theta, visuals)

    print("-----------------------------------")
    target_T3 = robot.get_pose_profile(target_q3)
    print(f"Target pose profile 3: {target_T3}")
    task2(robot, T, dt, q0, u, control_t, target_q3, Kp, Ki, Kd, Kp_theta, Ki_theta, Kd_theta, visuals)

    print("-----------------------------------")
    target_T4 = robot.get_pose_profile(target_q4)
    print(f"Target pose profile 4: {target_T4}")
    task2(robot, T, dt, q0, u, control_t, target_q4, Kp, Ki, Kd, Kp_theta, Ki_theta, Kd_theta, visuals)
    print("-----------------------------------")


if __name__ == "__main__":

    ## Franka robot object
    robot = fr.Franka()
    model = robot.model # Pinocchio model

    ## Simulation parameters
    # Time period
    T = 10
    # Time step
    dt = 0.001
    
    ## Initial state
    # Joints configuration
    pin.seed(10837383)
    # q0 = pin.randomConfiguration(model)
    q0 = pin.neutral(model)
    # Velocity
    u = np.zeros(model.nv)

    ## Initial Control torque
    control_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ## PID gains
    Kp = 50.
    Ki = 0.0
    Kd = 1.0

    Kp_theta = 56.
    Ki_theta = 0.
    Kd_theta = 0.

    # Task 1
    # task1(robot, T, dt, q0, u, control_t, visuals=True)

    # Task 2
    target_q = np.array([1.0, -0.645, -1.65, -2.15, -2.31,  2.18, 0.3])
    # try:
    #     task2(robot, T, dt, q0, u, control_t, target_q, Kp, Ki, Kd, Kp_theta)
    # except KeyboardInterrupt:
    #     exit()

    # Task 3
    # Target pose profiles
    target_q1 = np.array([0.6, -0.645, -0.65, -0.15, -0.31,  0.18, 0.3])
    target_q2 = np.array([0.5, -0.645, -1.65, -2.15, -2.31,  2.18, 0.3])
    target_q3 = np.array([1.0, -0.645, -1.65, -2.15, -2.31,  2.18, 0.3])
    target_q4 = np.array([0.5, -0.645, -1.65, -2.15, -2.31,  2.18, 0.3])

    
    try:
        task3(robot, T, dt, q0, u, control_t, target_q1, target_q2, target_q3, target_q4, Kp, Ki, Kd)
    except KeyboardInterrupt:
        exit()