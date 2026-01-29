import os
import csv
import datetime
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from episodes.roll import EPISODE_NAME, SIM_PARAMS
from quat import quat_to_euler

# Import the run_episode functions from both simulators
from run_jfsim import run_episode as run_jfsim_episode
from run_jsbsim import run_episode as run_jsbsim_episode

def visualize_comparison(jf_history, jf_time, jsb_history, jsb_time, png_path):
    """
    Plots JFSim vs JSBSim results on the same axes.
    """
    # Calculate Euler Angles for JFSim
    v_quat_to_euler = jax.vmap(quat_to_euler)
    jf_euler = v_quat_to_euler(jnp.array(jf_history.quat)) * (180.0 / jnp.pi)
    
    # Calculate Euler Angles for JSBSim
    jsb_euler = v_quat_to_euler(jnp.array(jsb_history.quat)) * (180.0 / jnp.pi)

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Comparison: JFSim vs JSBSim\nEpisode: {EPISODE_NAME}', fontsize=16)

    # 1. Position
    axs[0, 0].plot(jf_time, jf_history.pos[:, 0], 'b-', label='JF: North')
    axs[0, 0].plot(jsb_time, jsb_history.pos[:, 0], 'b--', label='JSB: North')
    axs[0, 0].plot(jf_time, jf_history.pos[:, 1], 'g-', label='JF: East')
    axs[0, 0].plot(jsb_time, jsb_history.pos[:, 1], 'g--', label='JSB: East')
    axs[0, 0].plot(jf_time, -jf_history.pos[:, 2], 'r-', label='JF: Alt')
    axs[0, 0].plot(jsb_time, -jsb_history.pos[:, 2], 'r--', label='JSB: Alt')
    axs[0, 0].set_title('Position (NED)')
    axs[0, 0].set_ylabel('Meters')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Velocity (Body)
    axs[0, 1].plot(jf_time, jf_history.vel[:, 0], 'b-', label='JF: u')
    axs[0, 1].plot(jsb_time, jsb_history.vel[:, 0], 'b--', label='JSB: u')
    axs[0, 1].plot(jf_time, jf_history.vel[:, 1], 'g-', label='JF: v')
    axs[0, 1].plot(jsb_time, jsb_history.vel[:, 1], 'g--', label='JSB: v')
    axs[0, 1].plot(jf_time, jf_history.vel[:, 2], 'r-', label='JF: w')
    axs[0, 1].plot(jsb_time, jsb_history.vel[:, 2], 'r--', label='JSB: w')
    axs[0, 1].set_title('Body Velocity')
    axs[0, 1].set_ylabel('m/s')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Attitude (Euler)
    axs[1, 0].plot(jf_time, jf_euler[:, 0], 'b-', label='JF: Roll')
    axs[1, 0].plot(jsb_time, jsb_euler[:, 0], 'b--', label='JSB: Roll')
    axs[1, 0].plot(jf_time, jf_euler[:, 1], 'g-', label='JF: Pitch')
    axs[1, 0].plot(jsb_time, jsb_euler[:, 1], 'g--', label='JSB: Pitch')
    axs[1, 0].plot(jf_time, jf_euler[:, 2], 'r-', label='JF: Yaw')
    axs[1, 0].plot(jsb_time, jsb_euler[:, 2], 'r--', label='JSB: Yaw')
    axs[1, 0].set_title('Attitude (Euler Angles)')
    axs[1, 0].set_ylabel('Degrees')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Angular Rates
    axs[1, 1].plot(jf_time, jf_history.omega[:, 0], 'b-', label='JF: p')
    axs[1, 1].plot(jsb_time, jsb_history.omega[:, 0], 'b--', label='JSB: p')
    axs[1, 1].plot(jf_time, jf_history.omega[:, 1], 'g-', label='JF: q')
    axs[1, 1].plot(jsb_time, jsb_history.omega[:, 1], 'g--', label='JSB: q')
    axs[1, 1].plot(jf_time, jf_history.omega[:, 2], 'r-', label='JF: r')
    axs[1, 1].plot(jsb_time, jsb_history.omega[:, 2], 'r--', label='JSB: r')
    axs[1, 1].set_title('Angular Rates')
    axs[1, 1].set_ylabel('rad/s')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(png_path)
    print(f"Comparison plot saved to {png_path}")

def save_to_csv(history, time_array, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = [
            "time", 
            "pos_n", "pos_e", "pos_d", 
            "vel_u", "vel_v", "vel_w", 
            "quat_0", "quat_1", "quat_2", "quat_3", 
            "omega_p", "omega_q", "omega_r"
        ]
        writer.writerow(header)
        for i in range(len(time_array)):
            row = [float(time_array[i])] + \
                  [float(x) for x in history.pos[i]] + \
                  [float(x) for x in history.vel[i]] + \
                  [float(x) for x in history.quat[i]] + \
                  [float(x) for x in history.omega[i]]
            writer.writerow(row)
    print(f"CSV saved to {csv_path}")

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = f"logs/comparison/{EPISODE_NAME}/{timestamp}"
    os.makedirs(base_log_dir, exist_ok=True)

    print("--- Running JFSim ---")
    jf_history, jf_time = run_jfsim_episode()
    
    print("\n--- Running JSBSim ---")
    jsb_history, jsb_time = run_jsbsim_episode()

    print("\n--- Saving Results ---")
    jf_csv = os.path.join(base_log_dir, "jfsim.csv")
    jsb_csv = os.path.join(base_log_dir, "jsbsim.csv")
    png_path = os.path.join(base_log_dir, "comparison.png")

    save_to_csv(jf_history, jf_time, jf_csv)
    save_to_csv(jsb_history, jsb_time, jsb_csv)
    visualize_comparison(jf_history, jf_time, jsb_history, jsb_time, png_path)

    print(f"\nComparison complete in {base_log_dir}")
