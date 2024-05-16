import numpy as np
import pinocchio as pin
import proxsuite

qp_init = False
prev_error = None
sum_error = 0.

def fk_all(model:pin.Model, data:pin.Data, q:np.ndarray):
    """
    Compute forward kinematics for all frames
    """
    pin.forwardKinematics(model, data, q) # FK
    pin.updateFramePlacements(model, data) # Update frames

def qp_control(model:pin.Model, data:pin.Data, qd:np.ndarray, q_k:np.ndarray, T_wd, frame_id, dt:float, Kp:float = 100., Kd:float = 1., Ki:float = 0.)->np.ndarray:
    """
    ProxSuite QP-based controller
    
    Args:
    model : Pinocchio model
    data : Pinocchio data
    qd : target joint configuration
    q_k : current joint configuration
    T_wd : desired transformation matrix
    frame_id : frame id
    dt : time step
    Kp : proportional gain
    Kd : derivative gain
    Ki : integral gain

    Returns:
    v : joint velocity
    """
    global qp
    global qp_init
    global prev_error
    global sum_error

    

    # first of all, let's compute the transformation matrix
    fk_all(model, data, qd)
    T_wd = data.oMf[frame_id].copy()

    qp_dim = model.nv
    qp_dim_eq = 0 # no equality constraint!
    qp_dim_in = model.nv # joint limits. ProxSuite supports double limits: d_min <= Cx <= d_max!
    qp = proxsuite.proxqp.dense.QP(qp_dim, qp_dim_eq, qp_dim_in)

    # first we need to compute FK
    fk_all(model, data, q_k)

    # We now need to get our current transformation matrix
    T_wb = data.oMf[frame_id]

    # We know compute the error
    error = pin.log(T_wb.actInv(T_wd)).vector
    if not qp_init:
        prev_error = np.copy(error)
    sum_error += (error * dt)
    error = Kp * error + Kd * (error - prev_error) / dt + Ki * sum_error
    prev_error = np.copy(error)

    # Compute Jacobian
    J = pin.computeFrameJacobian(model, data, q_k, frame_id, pin.ReferenceFrame.LOCAL) # Jacobian in local frame

    # Let's compute the QP matrices
    Q = J.T @ J
    q = -J.T @ error
    C = np.eye(model.nv) * dt
    d_min = model.lowerPositionLimit - q_k
    d_max = model.upperPositionLimit - q_k
    if not qp_init: # in first iteration we initialize the model
        qp.init(Q, q, None, None, C, d_min, d_max)
    else: # otherwise, we update the model
        qp.update(Q, q, None, None, C, d_min, d_max)
        qp_init = True

    # Let's solve the QP
    qp.solve()

    # We get back the results
    v = np.copy(qp.results.x)
    
    return v

def pid_torque_control(model:pin.Model, data:pin.Data, target_T:np.ndarray, current_q:np.ndarray, dt:float, Kp:float = 120., Ki:float = 0., Kd:float = 0.1)->tuple[np.ndarray, np.ndarray]:
    """
    PID torque controller

    Args:
    - model : Pinocchio model
    - data : Pinocchio data
    - target_T : target pose profile
    - current_q : current joint configuration
    - dt : time step
    - Kp : proportional gain
    - Ki : integral gain
    - Kd : derivative gain
    
    Returns:
    - control_t : control torque
    - error : error
    """
    global prev_error
    global sum_error

    frame_id = model.getFrameId("panda_ee")

    # Compute current transformation matrix
    fk_all(model, data, current_q)
    current_T = data.oMf[frame_id].copy()
    
    # print(f"Current T: {current_T}"
    #       f"Target T: {target_T}")

    # Compute error
    error = pin.log(current_T.actInv(target_T)).vector
    # print(f"Error: {error}")

    if prev_error is None:
        prev_error = np.copy(error)
    
    sum_error += (error * dt)
    diff_error = (error - prev_error) / dt

    error = Kp * error + Kd * diff_error + Ki * sum_error

    prev_error = np.copy(error)

    # Compute Jacobian
    J = pin.computeFrameJacobian(model, data, current_q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED) # Jacobian in world frame

    # Compute control torque
    control_t = J.T @ error

    return control_t, error