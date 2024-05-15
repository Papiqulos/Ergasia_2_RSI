from modeling import *
import numpy as np
  
def task1(): 

    robot, model, data, geometry_model, geometry_data = load_franka()

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

    qs, end_state = simulate(model, data, q0, u , control_t, T, dt, False)
    q, u = end_state
    print(f"\t\t\tEnd state\nposistion: {end_state[0]}\nvelocity: {end_state[1]}")
    
    # Visualize the robot
    # visualize(robot, qs)
    # while True: continue

def task2():
    robot, model, data, geometry_model, geometry_data = load_franka()

    T = 10
    dt = 0.1
    
    
    ## Initial state
    # Position
    # pin.seed(1083738)
    # q0 = pin.randomConfiguration(model)
    q0 = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])
    # q0 = np.array([ 1.13380891, -0.56585662, -2.65877219, -2.05434817, -2.81552492,  0.18592432, 0.28310915])

    # Velocity
    u = np.zeros(model.nv)

    # Control torque
    control_t = np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Target configuration
    target_q = np.array([ 1.13380891, -0.56585662, -2.65877219, -2.05434817, -2.81552492,  0.18592432, 0.28310915])

    qs, end_state = simulate(model, data, q0, u , control_t, T, dt, True, target_q)
    q, u = end_state
    print(f"\t\t\tEnd state\nEnd posistion: {end_state[0]}\nTarget position: {target_q}")

def task3():    
    pass


if __name__ == "__main__":

    task2()