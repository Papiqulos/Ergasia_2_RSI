from modeling import *
  
def task1(): 

    robot, model, data, geometry_model, geometry_data = load_franka()

    T = 0.1
    dt = 0.001
    
    
    ## Initial state
    # Position
    pin.seed(1083738)
    q0 = pin.randomConfiguration(model)
    # q0 = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])

    # Velocity
    u = np.zeros(model.nv)

    # Control torque
    control_t = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    qs, end_state = simulate(model, data, q0, u , control_t, T, dt)
    print(f"End state: {end_state}")

    q, u = end_state
    visualize(robot, qs)
    while True: continue

def task2():
    pass

def task3():    
    pass
    

if __name__ == "__main__":

    task1()