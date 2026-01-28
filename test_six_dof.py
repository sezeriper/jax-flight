import jax.numpy as jnp
from six_dof_aircraft import quat_rotate, quat_conjugate, calculate_forces_moments, State
from parameters import AIRCRAFT_PARAMS, ENV_PARAMS

def run_unit_tests():
    print("Running Unit Tests...")
    
    # Test 1: Identity Rotation
    v = jnp.array([1.0, 2.0, 3.0])
    q_id = jnp.array([1.0, 0.0, 0.0, 0.0])
    v_rot = quat_rotate(q_id, v)
    assert jnp.allclose(v, v_rot), f"Identity rotation failed: {v_rot}"
    print("  [Pass] Identity Rotation")
    
    # Test 2: Conjugate Rotation Reversal
    # Rotate by 45 deg about X, then rotate back by conjugate
    q_test = jnp.array([0.9238795, 0.3826834, 0.0, 0.0])
    v_rot = quat_rotate(q_test, v)
    v_back = quat_rotate(quat_conjugate(q_test), v_rot)
    assert jnp.allclose(v_back, v, atol=1e-5), f"Conjugate rotation failed: {v_back}"
    print("  [Pass] Conjugate Rotation")
    
    # Test 3: Lateral Stability
    # If we have positive sideslip (wind coming from right), we expect:
    # 1. Negative Side Force (pushing left)
    # 2. Positive Yaw Moment (Weathercock stability -> turn nose right to align with wind)
    # 3. Negative Roll Moment (Dihedral -> roll left/away from wind)
    state_slip = State(
        pos=jnp.array([0., 0., -100.]),
        vel=jnp.array([20.0, 5.0, 0.0]), # 5 m/s sideslip to the right
        quat=jnp.array([1., 0., 0., 0.]),
        omega=jnp.array([0., 0., 0.])
    )
    f, m = calculate_forces_moments(state_slip, AIRCRAFT_PARAMS, ENV_PARAMS)
    
    assert f[1] < 0, f"Side force should oppose sideslip. Got {f[1]}"
    assert m[2] > 0, f"Yaw moment should align with wind (weathercock). Got {m[2]}"
    assert m[0] < 0, f"Roll moment should be negative (stable dihedral). Got {m[0]}"
    print("  [Pass] Lateral Stability checks")
    
    print("All tests passed.\n")

if __name__ == "__main__":
    run_unit_tests()
