import jax.numpy as jnp

# ==============================================================================
# Environment Parameters
# ==============================================================================
ENV_PARAMS = {
    'g': 9.81,               # m/s^2
    'rho': 1.225,            # Air density (kg/m^3)
}

# ==============================================================================
# Aircraft Parameters (Standard small UAV)
# ==============================================================================
AIRCRAFT_PARAMS = {
    'mass': 13.5,            # kg
    'S': 0.55,               # Wing area (m^2)
    'b': 2.8956,             # Wingspan (m)
    'c': 0.18994,            # Mean Chord (m)
    'J': jnp.diag(jnp.array([0.1825, 0.2175, 0.2175])), # Inertia Tensor
    'J_inv': jnp.diag(1.0 / jnp.array([0.1825, 0.2175, 0.2175])),
    
    # Aerodynamic Coefficients (Simplified Linear Model)
    'C_L_0': 0.28,           # Lift at zero alpha
    'C_L_alpha': 3.45,       # Lift slope
    'C_D_0': 0.03,           # Parasitic drag
    'C_D_alpha': 0.30,       # Drag induced by alpha
    'C_m_0': 0.0,            # Pitch moment at zero alpha
    'C_m_alpha': -0.38,      # Pitch stability (negative = stable)
    'C_m_q': -3.6,           # Pitch damping
    
    # Lateral Aerodynamics
    'C_Y_beta': -0.98,       # Side force due to sideslip
    'C_l_beta': -0.12,       # Dihedral effect (Roll stability)
    'C_l_p': -0.26,          # Roll damping
    'C_l_r': 0.14,           # Roll due to yaw rate
    'C_n_beta': 0.25,        # Weathercock stability (Yaw stability)
    'C_n_p': 0.022,          # Yaw due to roll rate
    'C_n_r': -0.35,          # Yaw damping

    # Control Derivatives
    'C_l_delta_a': 0.08,     # Roll due to aileron
    'C_n_delta_a': 0.06,     # Yaw due to aileron (Adverse yaw)
    'C_m_delta_e': -0.5,     # Pitch due to elevator
    'C_Y_delta_r': 0.17,     # Side force due to rudder
    'C_l_delta_r': 0.105,    # Roll due to rudder
    'C_n_delta_r': -0.032,   # Yaw due to rudder
}
