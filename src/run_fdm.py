import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from episodes.roll import EPISODE_NAME, ENV_PARAMS, SIM_PARAMS, INIT_PARAMS, create_controls  # Import parameters
from aircrafts.simple_uav import AIRCRAFT_PARAMS
import csv
import datetime
import os
from fdm import run_simulation_loop, State, Controls
from quat import quat_to_euler

# ==============================================================================
# Visualization Logic
# ==============================================================================
def visualize_results(history: State, time: jnp.ndarray, T_total: float, png_path: str):
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
    plt.savefig(png_path)

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # 1. Simulation settings
    T_TOTAL = SIM_PARAMS['T_total']
    DT = SIM_PARAMS['dt']
    STEPS = int(T_TOTAL / DT)
    
    # Initial Conditions
    # Flying North at 25 m/s, Level flight, Altitude -100m (Up is negative Z)
    init_state = State(
        pos=INIT_PARAMS['pos'],
        vel=INIT_PARAMS['vel'],
        quat=INIT_PARAMS['quat'],
        omega=INIT_PARAMS['omega']
    )

    # 2. Controls
    controls = create_controls(STEPS, DT)
    
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
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"jfsim_{EPISODE_NAME}_{timestamp}"
    csv_filename = filename + ".csv"
    png_filename = filename + ".png"
    csv_path = f"logs/jfsim/{EPISODE_NAME}/{timestamp}/{csv_filename}"
    png_path = f"logs/jfsim/{EPISODE_NAME}/{timestamp}/{png_filename}"

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 3. Visualize
    visualize_results(history, time_array, T_TOTAL, png_path)

    # 4. Save to CSV
    print(f"Saving results to {csv_path}...")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Header
        header = [
            "time", 
            "pos_n", "pos_e", "pos_d", 
            "vel_u", "vel_v", "vel_w", 
            "quat_0", "quat_1", "quat_2", "quat_3", 
            "omega_p", "omega_q", "omega_r"
        ]
        writer.writerow(header)
        
        # Data
        # Converting JAX arrays to numpy for iteration
        times = jnp.array(time_array)
        pos = jnp.array(history.pos)
        vel = jnp.array(history.vel)
        quat = jnp.array(history.quat)
        omega = jnp.array(history.omega)
        
        for i in range(len(times)):
            row = [
                times[i].item(),
                pos[i, 0].item(), pos[i, 1].item(), pos[i, 2].item(),
                vel[i, 0].item(), vel[i, 1].item(), vel[i, 2].item(),
                quat[i, 0].item(), quat[i, 1].item(), quat[i, 2].item(), quat[i, 3].item(),
                omega[i, 0].item(), omega[i, 1].item(), omega[i, 2].item()
            ]
            writer.writerow(row)
            
    print("CSV saved successfully.")