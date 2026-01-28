import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import matplotlib.pyplot as plt
from typing import NamedTuple, Dict
from parameters import ENV_PARAMS, AIRCRAFT_PARAMS  # Import parameters

# ==============================================================================
# 1. State Definition (Pytree)
# ==============================================================================
class State(NamedTuple):
    pos: jnp.ndarray   # [Pn, Pe, Pd] (Inertial Frame)
    vel: jnp.ndarray   # [u, v, w]    (Body Frame)
    quat: jnp.ndarray  # [q0, q1, q2, q3] (Body to Inertial, Scalar First)
    omega: jnp.ndarray # [p, q, r]    (Body Frame)

class Controls(NamedTuple):
    da: float = 0.0 # Aileron (rad)
    de: float = 0.0 # Elevator (rad)
    dr: float = 0.0 # Rudder (rad)
    dt: float = 0.0 # Throttle (0-1)

# ==============================================================================
# 2. Helper Functions
# ==============================================================================
@jit
def quat_rotate(q, v):
    """Rotates vector v by quaternion q.
    Rotates from Body -> Inertial if q is q_body_to_inertial.
    """
    # Formula: v_new = v + 2 * cross(q_xyz, cross(q_xyz, v) + q_w * v)
    q0, q1, q2, q3 = q
    q_vec = jnp.array([q1, q2, q3])
    t = 2.0 * jnp.cross(q_vec, v)
    return v + q0 * t + jnp.cross(q_vec, t)

@jit
def quat_conjugate(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

@jit
def quat_deriv(q, omega):
    """
    Computes q_dot = 0.5 * Omega * q
    Convention: q = [q0, q1, q2, q3] (Scalar first)
                omega = [p, q, r] (Body rates)
    """
    p, q_rate, r = omega  # Explicit naming to avoid confusion
    
    # Matrix representation of quaternion multiplication
    Omega = jnp.array([
        [0.0,   -p,     -q_rate, -r],
        [p,      0.0,    r,      -q_rate],
        [q_rate, -r,     0.0,     p],
        [r,      q_rate, -p,      0.0]
    ])
    return 0.5 * jnp.dot(Omega, q)

@jit
def quat_to_euler(q):
    """Converts Quaternion to Roll, Pitch, Yaw (for plotting only)"""
    w, x, y, z = q
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = jnp.where(jnp.abs(sinp) >= 1,
                      jnp.sign(sinp) * (jnp.pi / 2),
                      jnp.arcsin(sinp))
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    return jnp.array([roll, pitch, yaw])

# ==============================================================================
# 3. Physics Engine (Aerodynamics + Dynamics)
# ==============================================================================
@jit
def calculate_forces_moments(state: State, controls: Controls, aircraft_params: Dict, env_params: Dict):
    """Computes aerodynamic forces and moments in Body Frame."""
    u, v, w = state.vel
    p, q, r = state.omega
    
    # Airspeed and Dynamic Pressure
    Va = jnp.sqrt(u**2 + v**2 + w**2)
    alpha = jnp.arctan2(w, u) # Angle of Attack (approx)
    
    # Protect against divide by zero if stationary (unlikely in flight sim)
    safe_Va = jnp.where(Va < 0.1, 0.1, Va)
    
    q_bar = 0.5 * env_params['rho'] * Va**2
    
    # 1. Aerodynamic Forces (3D Wind Frame to Body Frame)
    # Wind-frame forces: X_w = -Drag, Y_w = SideForce, Z_w = -Lift
    cl = aircraft_params['C_L_0'] + aircraft_params['C_L_alpha'] * alpha
    cd = aircraft_params['C_D_0'] + aircraft_params['C_D_alpha'] * alpha**2
    
    # beta is angle between velocity vector and x-z plane of body
    beta = jnp.arcsin(v / safe_Va)
    cy = aircraft_params['C_Y_beta'] * beta + aircraft_params['C_Y_delta_r'] * controls.dr

    
    # Precompute trig terms for the 3D rotation matrix
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    
    # Transform Lift, Drag, and Side Force to Body Frame
    # Formula: F_body = R_y(alpha) * R_z(-beta) * [-Drag, SideForce, -Lift]^T
    f_aero_x = q_bar * aircraft_params['S'] * (-cd * ca * cb + cy * ca * sb + cl * sa)
    f_aero_y = q_bar * aircraft_params['S'] * (-cd * sb + cy * cb)
    f_aero_z = q_bar * aircraft_params['S'] * (-cd * sa * cb + cy * sa * sb - cl * ca)
    
    F_aero = jnp.array([f_aero_x, f_aero_y, f_aero_z])
    
    # 2. Aerodynamic Moments (Simplified)
    # Pitch moment with damping
    cm = aircraft_params['C_m_0'] + aircraft_params['C_m_alpha'] * alpha + aircraft_params['C_m_q'] * (aircraft_params['c']/(2*safe_Va)) * q + aircraft_params['C_m_delta_e'] * controls.de
    m_pitch = q_bar * aircraft_params['S'] * aircraft_params['c'] * cm
    
    # Roll and Yaw Moments (Lateral/Directional)
    # Normalized rates
    p_norm = (aircraft_params['b'] * p) / (2 * safe_Va)
    r_norm = (aircraft_params['b'] * r) / (2 * safe_Va)
    
    cl_moment = aircraft_params['C_l_beta'] * beta + aircraft_params['C_l_p'] * p_norm + aircraft_params['C_l_r'] * r_norm + aircraft_params['C_l_delta_a'] * controls.da + aircraft_params['C_l_delta_r'] * controls.dr
    cn_moment = aircraft_params['C_n_beta'] * beta + aircraft_params['C_n_p'] * p_norm + aircraft_params['C_n_r'] * r_norm + aircraft_params['C_n_delta_a'] * controls.da + aircraft_params['C_n_delta_r'] * controls.dr
    
    m_roll = q_bar * aircraft_params['S'] * aircraft_params['b'] * cl_moment
    m_yaw = q_bar * aircraft_params['S'] * aircraft_params['b'] * cn_moment
    
    M_aero = jnp.array([m_roll, m_pitch, m_yaw])
    
    return F_aero, M_aero

@jit
def equations_of_motion(state: State, controls: Controls, aircraft_params: Dict, env_params: Dict) -> State:
    # Unpack
    v_b = state.vel
    q_b = state.quat
    w_b = state.omega
    mass = aircraft_params['mass']
    
    # --- Forces ---
    # Gravity (Rotated from NED to Body)
    g_vec = jnp.array([0.0, 0.0, env_params['g']])
    f_gravity = quat_rotate(quat_conjugate(q_b), g_vec) * mass
    
    # Aerodynamics
    f_aero, m_aero = calculate_forces_moments(state, controls, aircraft_params, env_params)
    
    # Total Force
    f_total = f_gravity + f_aero # + Thrust (assumed 0 for glide demo)
    
    # --- Dynamics (Newton-Euler) ---
    # Translational: v_dot = F/m - (omega x v)
    v_dot = (f_total / mass) - jnp.cross(w_b, v_b)
    
    # Rotational: w_dot = J_inv * (M - (omega x J*omega))
    gyroscopic = jnp.cross(w_b, jnp.dot(aircraft_params['J'], w_b))
    w_dot = jnp.dot(aircraft_params['J_inv'], (m_aero - gyroscopic))
    
    # --- Kinematics ---
    # Position: p_dot = R * v
    p_dot = quat_rotate(q_b, v_b)
    
    # Quaternion: q_dot
    q_dot = quat_deriv(q_b, w_b)
    
    return State(pos=p_dot, vel=v_dot, quat=q_dot, omega=w_dot)

# ==============================================================================
# 4. Integration (RK4)
# ==============================================================================
@jit
def rk4_step(state, controls, dt, aircraft_params, env_params):
    
    k1 = equations_of_motion(state, controls, aircraft_params, env_params)
    
    s2 = jax.tree_util.tree_map(lambda x, k: x + 0.5*dt*k, state, k1)
    k2 = equations_of_motion(s2, controls, aircraft_params, env_params)
    
    s3 = jax.tree_util.tree_map(lambda x, k: x + 0.5*dt*k, state, k2)
    k3 = equations_of_motion(s3, controls, aircraft_params, env_params)
    
    s4 = jax.tree_util.tree_map(lambda x, k: x + dt*k, state, k3)
    k4 = equations_of_motion(s4, controls, aircraft_params, env_params)
    
    # Combine
    new_state = jax.tree_util.tree_map(
        lambda x, d1, d2, d3, d4: x + (dt/6.0)*(d1 + 2*d2 + 2*d3 + d4),
        state, k1, k2, k3, k4
    )
    
    # Normalization Constraint (Critical for Quaternions)
    q_norm = new_state.quat / jnp.linalg.norm(new_state.quat)
    
    # Re-pack into NamedTuple
    return State(new_state.pos, new_state.vel, q_norm, new_state.omega)

# ==============================================================================
# 5. Core Simulation Logic (Pure JIT)
# ==============================================================================
@partial(jit, static_argnames=['steps'])
def run_simulation_loop(init_state: State, init_controls: Controls, aircraft_params: Dict, env_params: Dict, dt: float, steps: int):
    """
    Pure JIT compiled simulation loop using jax.lax.scan.
    Args:
        init_controls: Can be a Controls object where each field is an array of length 'steps' (for pre-programmed controls)
                       OR a Checks object of scalars (for constant control).
                       However, for scan, we usually want time-varying.
                       Let's assume inputs 'controls' are passed as a Struct of Arrays matching 'steps'.
    """
    def step_fn(carry_state, control_input):
        next_state = rk4_step(carry_state, control_input, dt, aircraft_params, env_params)
        # Return next_state as carry AND as element of history (post-update)
        return next_state, next_state

    # Use lax.scan for the loop. 
    # control_input will be sliced from init_controls at each step
    final_state, history = jax.lax.scan(step_fn, init_state, init_controls, length=steps)
    
    return history

# ==============================================================================
# 6. Visualization Logic
# ==============================================================================
def visualize_results(history: State, time: jnp.ndarray, T_total: float):
    """
    Handles all plotting and visualization logic using Matplotlib.
    This runs on CPU and interprets the JAX results.
    """
    # Calculate Euler Angles for plotting
    v_quat_to_euler = jax.vmap(quat_to_euler)
    euler_angles = v_quat_to_euler(history.quat) * (180.0 / jnp.pi) # To Degrees
    
    # --- PLOTTING ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'6-DOF JAX Simulation ({T_total}s Glider Demo)')
    
    # 1. Position
    axs[0, 0].plot(time, history.pos[:, 0], label='North (x)')
    axs[0, 0].plot(time, history.pos[:, 1], label='East (y)')
    axs[0, 0].plot(time, -history.pos[:, 2], label='Altitude (-z)') # Plot Altitude positive
    axs[0, 0].set_title('Position (NED)')
    axs[0, 0].set_ylabel('Meters')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 2. Velocity (Body)
    axs[0, 1].plot(time, history.vel[:, 0], label='u (fwd)')
    axs[0, 1].plot(time, history.vel[:, 1], label='v (right)')
    axs[0, 1].plot(time, history.vel[:, 2], label='w (down)')
    axs[0, 1].set_title('Body Velocity')
    axs[0, 1].set_ylabel('m/s')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # 3. Attitude (Euler)
    axs[1, 0].plot(time, euler_angles[:, 0], label='Roll')
    axs[1, 0].plot(time, euler_angles[:, 1], label='Pitch')
    axs[1, 0].plot(time, euler_angles[:, 2], label='Yaw')
    axs[1, 0].set_title('Attitude (Euler Angles)')
    axs[1, 0].set_ylabel('Degrees')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # 4. Angular Rates
    axs[1, 1].plot(time, history.omega[:, 0], label='p (Roll Rate)')
    axs[1, 1].plot(time, history.omega[:, 1], label='q (Pitch Rate)')
    axs[1, 1].plot(time, history.omega[:, 2], label='r (Yaw Rate)')
    axs[1, 1].set_title('Angular Rates')
    axs[1, 1].set_ylabel('rad/s')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 7. Main Execution
# ==============================================================================
if __name__ == "__main__":
    # 1. Simulation settings
    T_TOTAL = 5.0
    DT = 0.01
    STEPS = int(T_TOTAL / DT)
    
    # Initial Conditions
    # Flying North at 25 m/s, Level flight, Altitude -100m (Up is negative Z)
    init_state = State(
        pos=jnp.array([0.0, 0.0, -100.0]), 
        vel=jnp.array([25.0, 0.0, 0.0]),   
        quat=jnp.array([1.0, 0.0, 0.0, 0.0]), # Identity quaternion
        omega=jnp.array([0.0, 0.0, 0.0])
    )

    # 2. Controls (Zero for now)
    controls = Controls(
        da=jnp.zeros(STEPS),
        de=jnp.zeros(STEPS),
        dr=jnp.zeros(STEPS),
        dt=jnp.zeros(STEPS)
    )
    
    print("Compiling and Running Simulation...")
    # Call the jitted simulation loop
    history_raw = run_simulation_loop(init_state, controls, AIRCRAFT_PARAMS, ENV_PARAMS, DT, STEPS)
    print("Simulation Complete.")

    # 2. Post-Process Data
    history = jax.tree_util.tree_map(
        lambda init, hist: jnp.concatenate([init[None, :], hist], axis=0), 
        init_state, 
        history_raw
    )
    
    # Create time array matching history length (0 to T inclusive)
    time_array = jnp.linspace(0, T_TOTAL, STEPS + 1)
    
    # 3. Visualize
    visualize_results(history, time_array, T_TOTAL)