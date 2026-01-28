
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import numpy as np
import jsbsim
import os
import sys

# Add current directory to path to import six_dof_aircraft
sys.path.append(os.getcwd())

from six_dof_aircraft import run_simulation_loop, State, Controls, visualize_results
from parameters import AIRCRAFT_PARAMS, ENV_PARAMS

def create_controls(steps, dt):
    # Roll pulse: 1s to 2s, magnitude 5 deg
    da = jnp.zeros(steps)
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

def run_jax_sim(dt, steps, controls):
    print("Running JAX Simulation...")
    init_state = State(
        pos=jnp.array([0.0, 0.0, -100.0]), 
        vel=jnp.array([25.0, 0.0, 0.0]),   
        quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
        omega=jnp.array([0.0, 0.0, 0.0])
    )
    # The run_simulation_loop is JIT-ed and expects JAX arrays in controls
    history_raw = run_simulation_loop(init_state, controls, AIRCRAFT_PARAMS, ENV_PARAMS, dt, steps)
    
    # Prepend initial state
    history = jax.tree_util.tree_map(
        lambda init, hist: jnp.concatenate([init[None, :], hist], axis=0), 
        init_state, 
        history_raw
    )
    return history

def run_jsbsim_sim(dt, steps, controls_jax):
    print("Running JSBSim Simulation...")
    fdm = jsbsim.FGFDMExec(os.getcwd())
    
    # Point to the local directory containing 'simple_uav/simple_uav.xml'
    fdm.set_aircraft_path("jsbsim_model")
    
    fdm.load_model("simple_uav")
    
    # Setup IC
    fdm.set_property_value("ic/lat-gc-deg", 0.0)
    fdm.set_property_value("ic/long-gc-deg", 0.0)
    fdm.set_property_value("ic/h-sl-ft", 100.0 * 3.28084) # 100m to ft
    
    # Velocity: 25 m/s Body X
    fdm.set_property_value("ic/u-fps", 25.0 * 3.28084)
    fdm.set_property_value("ic/v-fps", 0.0)
    fdm.set_property_value("ic/w-fps", 0.0)
    
    # Attitude: Level
    fdm.set_property_value("ic/phi-deg", 0.0)
    fdm.set_property_value("ic/theta-deg", 0.0)
    fdm.set_property_value("ic/psi-true-deg", 0.0)
    
    # Rates
    fdm.set_property_value("ic/p-rad_sec", 0.0)
    fdm.set_property_value("ic/q-rad_sec", 0.0)
    fdm.set_property_value("ic/r-rad_sec", 0.0)
    
    # Initialize
    fdm.set_dt(dt)
    fdm.run_ic()
    
    # Turn off engines (it's a glider)
    fdm.set_property_value("propulsion/set-running", 0)
    
    # Data storage
    jsb_data = {
        'time': [],
        'pos_n': [], 'pos_e': [], 'pos_d': [],
        'u': [], 'v': [], 'w': [],
        'phi': [], 'theta': [], 'psi': [],
        'p': [], 'q': [], 'r': [],
        'beta': []
    }
    
    # Initial Lat/Lon for relative position
    lat0 = fdm.get_property_value("ic/lat-gc-rad")
    lon0 = fdm.get_property_value("ic/long-gc-rad")
    re_ft = 20925646.3 # Approx Earth radius in ft
    
    # Extract numpy arrays from checks for faster access if needed, or iterate
    # controls_jax has shape (steps,)
    da_arr = np.array(controls_jax.da)
    de_arr = np.array(controls_jax.de)
    dr_arr = np.array(controls_jax.dr)
    
    for i in range(steps + 1):
        # Apply Controls (if i < steps)
        if i < steps:
            fdm.set_property_value("fcs/left-aileron-pos-rad", float(da_arr[i]))
            fdm.set_property_value("fcs/right-aileron-pos-rad", -float(da_arr[i])) # Symmetric
            fdm.set_property_value("fcs/elevator-pos-rad", float(de_arr[i]))
            fdm.set_property_value("fcs/rudder-pos-rad", float(dr_arr[i]))
        
        # Store Data
        jsb_data['time'].append(fdm.get_sim_time())
        
        # Position
        lat = fdm.get_property_value("position/lat-gc-rad")
        lon = fdm.get_property_value("position/long-gc-rad")
        alt = fdm.get_property_value("position/h-sl-meters")
        
        dn = (lat - lat0) * re_ft * 0.3048 # ft to meters
        de = (lon - lon0) * re_ft * 0.3048 * np.cos(lat0)
        
        jsb_data['pos_n'].append(dn)
        jsb_data['pos_e'].append(de)
        jsb_data['pos_d'].append(-alt) 
        
        # Velocity (Body) m/s
        jsb_data['u'].append(fdm.get_property_value("velocities/u-fps") * 0.3048)
        jsb_data['v'].append(fdm.get_property_value("velocities/v-fps") * 0.3048)
        jsb_data['w'].append(fdm.get_property_value("velocities/w-fps") * 0.3048)
        
        # Attitude (Rad to Deg)
        jsb_data['phi'].append(fdm.get_property_value("attitude/phi-deg"))
        jsb_data['theta'].append(fdm.get_property_value("attitude/theta-deg"))
        jsb_data['psi'].append(fdm.get_property_value("attitude/psi-deg"))
        
        # Rates (Rad/s)
        jsb_data['p'].append(fdm.get_property_value("velocities/p-rad_sec"))
        jsb_data['q'].append(fdm.get_property_value("velocities/q-rad_sec"))
        jsb_data['r'].append(fdm.get_property_value("velocities/r-rad_sec"))
        
        # Aero
        jsb_data['beta'].append(fdm.get_property_value("aero/beta-rad"))
        
        n_mom = fdm.get_property_value("moments/n-aero-lbsft")
        # r_val = fdm.get_property_value("velocities/r-rad_sec")
        # if i % 100 == 0:
        #      print(f"T={fdm.get_sim_time():.2f} Beta={jsb_data['beta'][-1]:.4f} N_mom={n_mom:.4f}")
        
        fdm.run()
        
    return jsb_data

def compare_results(jax_res, jsb_res, time_arr):
    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle('Sim Comparison: JAX (Blue) vs JSBSim (Orange)')
    
    # 1. Position
    axs[0, 0].plot(time_arr, jax_res.pos[:, 0], 'b-', label='JAX N')
    axs[0, 0].plot(time_arr, jax_res.pos[:, 1], 'b--', label='JAX E')
    axs[0, 0].plot(time_arr, -jax_res.pos[:, 2], 'b:', label='JAX Alt') # Plot Alt
    
    axs[0, 0].plot(jsb_res['time'], jsb_res['pos_n'], 'r-', label='JSB N')
    axs[0, 0].plot(jsb_res['time'], jsb_res['pos_e'], 'r--', label='JSB E')
    axs[0, 0].plot(jsb_res['time'], [-x for x in jsb_res['pos_d']], 'r:', label='JSB Alt')
    axs[0, 0].set_title('Position')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 2. Velocity
    axs[0, 1].plot(time_arr, jax_res.vel[:, 0], 'b-', label='JAX u')
    axs[0, 1].plot(time_arr, jax_res.vel[:, 1], 'b--', label='JAX v')
    axs[0, 1].plot(time_arr, jax_res.vel[:, 2], 'b:', label='JAX w')
    
    axs[0, 1].plot(jsb_res['time'], jsb_res['u'], 'r-', label='JSB u')
    axs[0, 1].plot(jsb_res['time'], jsb_res['v'], 'r--', label='JSB v')
    axs[0, 1].plot(jsb_res['time'], jsb_res['w'], 'r:', label='JSB w')
    axs[0, 1].set_title('Body Velocity')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Calculate Euler and Beta for JAX
    from six_dof_aircraft import quat_to_euler
    v_quat_to_euler = jax.vmap(quat_to_euler)
    jax_euler = v_quat_to_euler(jax_res.quat) * (180.0 / jnp.pi)
    
    # JAX Beta
    # beta = asin(v / Va)
    Va = jnp.linalg.norm(jax_res.vel, axis=1)
    jax_beta = jnp.arcsin(jax_res.vel[:, 1] / jnp.where(Va < 0.1, 0.1, Va))
    
    # 3. Attitude (Euler)
    axs[1, 0].plot(time_arr, jax_euler[:, 0], 'b-', label='JAX Roll')
    axs[1, 0].plot(time_arr, jax_euler[:, 1], 'b--', label='JAX Pitch')
    axs[1, 0].plot(time_arr, jax_euler[:, 2], 'b:', label='JAX Yaw')
    
    axs[1, 0].plot(jsb_res['time'], jsb_res['phi'], 'r-', label='JSB Roll')
    axs[1, 0].plot(jsb_res['time'], jsb_res['theta'], 'r--', label='JSB Pitch')
    axs[1, 0].plot(jsb_res['time'], jsb_res['psi'], 'r:', label='JSB Yaw')
    axs[1, 0].set_title('Attitude (Deg)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # 4. Rates
    axs[1, 1].plot(time_arr, jax_res.omega[:, 0], 'b-', label='JAX p')
    axs[1, 1].plot(time_arr, jax_res.omega[:, 1], 'b--', label='JAX q')
    axs[1, 1].plot(time_arr, jax_res.omega[:, 2], 'b:', label='JAX r')
    
    axs[1, 1].plot(jsb_res['time'], jsb_res['p'], 'r-', label='JSB p')
    axs[1, 1].plot(jsb_res['time'], jsb_res['q'], 'r--', label='JSB q')
    axs[1, 1].plot(jsb_res['time'], jsb_res['r'], 'r:', label='JSB r')
    axs[1, 1].set_title('Body Rates')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # 5. Beta (Sideslip)
    axs[2, 0].plot(time_arr, jax_beta * 180.0/np.pi, 'b-', label='JAX Beta')
    axs[2, 0].plot(jsb_res['time'], np.array(jsb_res['beta']) * 180.0/np.pi, 'r--', label='JSB Beta')
    axs[2, 0].set_title('Sideslip (Beta, Deg)')
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    # 6. Yaw Zoom
    axs[2, 1].plot(time_arr, jax_euler[:, 2], 'b-', label='JAX Yaw')
    axs[2, 1].plot(jsb_res['time'], jsb_res['psi'], 'r--', label='JSB Yaw')
    axs[2, 1].set_title('Yaw Angle Zoom')
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_plot_pulse.png')
    print("Plot saved to comparison_plot_pulse.png")
    
    # Debug Print
    print(f"Final JAX Yaw: {jax_euler[-1, 2]:.4f}")
    print(f"Final JSB Yaw: {jsb_res['psi'][-1]:.4f}")
    print(f"Final JAX Beta: {jax_beta[-1]*180/np.pi:.4f}")
    print(f"Final JSB Beta: {jsb_res['beta'][-1]*180/np.pi:.4f}")
    
    # Try fetching moment property from fdm object? We can't access fdm here.
    # We should have stored it if we wanted to see it.
    # But for now let's rely on the previous print outputs or re-run if needed.
    # Actually, let's fix the run_jsbsim to allow printing.
    pass

if __name__ == "__main__":
    T_TOTAL = 5.0
    DT = 0.01
    STEPS = int(T_TOTAL / DT)
    time_array = jnp.linspace(0, T_TOTAL, STEPS + 1)
    
    controls = create_controls(STEPS, DT)
    
    jax_res = run_jax_sim(DT, STEPS, controls)
    jsb_res = run_jsbsim_sim(DT, STEPS, controls)
    
    compare_results(jax_res, jsb_res, time_array)
