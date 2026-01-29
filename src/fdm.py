import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Dict, NamedTuple
from quat import quat_rotate, quat_conjugate, quat_deriv

# ==============================================================================
# State Definition (Pytree)
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
# Physics Engine (Aerodynamics + Dynamics)
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
