import numpy as np
import FrankaRobot as fr
import pinocchio as pin
  
def task1(robot): 

    model = robot.model
    T = 10
    dt = 0.01
    
    
    ## Initial state
    # Position
    # pin.seed(1083738)
    q0 = pin.randomConfiguration(model)
    # q0 = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])
    # q0 = np.array([ 1.13380891, -0.56585662, -2.65877219, -2.05434817, -2.81552492,  0.18592432, 0.28310915])

    # Velocity
    u = np.zeros(model.nv)

    # Control torque
    control_t = np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    qs, end_state = robot.simulate(q0, u, control_t, T, dt)
    q, u = end_state
    print(f"\t\t\tEnd state\nposistion: {end_state[0]}\nvelocity: {end_state[1]}")
    
    # Visualize the robot
    # visualize(robot, qs)
    # while True: continue

def task2(robot):

    model = robot.model

    T = 10
    dt = 0.01
    
    ## Initial state
    # Position
    # pin.seed(1083738)
    # q0 = pin.randomConfiguration(model)
    q0 = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])
    # q0 = np.array([ 1.13380891, -0.56585662, -2.65877219, -2.05434817, -2.81552492,  0.18592432, 0.28310915])

    # Velocity
    u = np.zeros(model.nv)

    ## Control torque
    control_t = np.array([0.0, 5.0, 0.0, 0.0, 4.0, 0.0, 0.0])
    
    # Target configuration
    target_q = np.array([ 1.13380891, -0.56585662, -2.65877219, -2.05434817, -2.81552492,  0.18592432, 0.28310915])

    Kp = 100.
    Ki = 1.
    Kd = 0.

    qs, end_state = robot.simulate(q0, u , control_t, T, dt, target_q, Kp, Ki, Kd)
    # qs, end_state = simulate(model, data, q0, u , control_t, T, dt, target_q, Kp, Ki, Kd)
    q, u = end_state

    print(f"\nEnd posistion:\t\tTarget position:")
    for i in range(len(q)):
        print(f"{q[i]:.3f}\t\t\t{target_q[i]:.3f}")
    
    # Visualize the robot
    robot.visualize(qs)
    while True: continue

def task3(robot):    
    pass


if __name__ == "__main__":

    robot = fr.Franka()

    task1(robot)

    # task2(robot)

    # task3(robot)