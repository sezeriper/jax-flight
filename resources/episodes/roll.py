import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Dict
from fdm import State, Controls

EPISODE_NAME = "roll"

# ==============================================================================
# Environment Parameters
# ==============================================================================
ENV_PARAMS = {
    'g': 9.81,               # m/s^2
    'rho': 1.225,            # Air density (kg/m^3)
}

# ==============================================================================
# Simulation Parameters
# ==============================================================================
SIM_PARAMS = {
    'T_total': 5.0,          # Total simulation time (s)
    'dt': 0.01,              # Time step (s)
}

# ==============================================================================
# Initial State Parameters
# ==============================================================================
INIT_PARAMS = {
    'pos': jnp.array([0.0, 0.0, -100.0]),       # [Nt, Et, Dt] (m)
    'vel': jnp.array([25.0, 0.0, 0.0]),         # [u, v, w] (m/s)
    'quat': jnp.array([1.0, 0.0, 0.0, 0.0]),    # [q0, q1, q2, q3]
    'omega': jnp.array([0.0, 0.0, 0.0]),        # [p, q, r] (rad/s)
}

def create_controls(steps, dt):
    # Roll pulse: 1s to 2s, magnitude 5 deg
    # create a pulse
    start_step = int(1.0 / dt)
    end_step = int(2.0 / dt)
    pulse_val = 5.0 * (np.pi / 180.0)
    
    # Use JAX index update or numpy 
    # Since we passing this to JIT, we can construct it with numpy before passing
    da_np = np.zeros(steps)
    da_np[start_step:end_step] = pulse_val
    
    return Controls(
        da=jnp.array(da_np),
        de=jnp.zeros(steps),
        dr=jnp.zeros(steps),
        dt=jnp.zeros(steps)
    )
