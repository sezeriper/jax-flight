import jax.numpy as jnp
from jax import jit

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