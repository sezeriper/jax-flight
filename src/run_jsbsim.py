import os
import csv
import datetime
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jsbsim
from episodes.roll import EPISODE_NAME, ENV_PARAMS, SIM_PARAMS, INIT_PARAMS, create_controls
from aircrafts.simple_uav import AIRCRAFT_PARAMS
from quat import quat_to_euler, quat_from_euler
from fdm import State

def euler_to_quat(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy
    return np.array([q0, q1, q2, q3])

# ==============================================================================
# JSBSim XML Generation
# ==============================================================================
def generate_jsbsim_xml(params, aircraft_name="simple_uav"):
    """
    Generates a JSBSim XML file based on the provided aircraft parameters.
    """
    ixx = float(params['J'][0, 0])
    iyy = float(params['J'][1, 1])
    izz = float(params['J'][2, 2])
    
    xml_content = f"""<?xml version="1.0"?>
<fdm_config name="{aircraft_name}" version="2.0" release="BETA">
    <fileheader>
        <author> Antigravity </author>
        <filecreationdate> {datetime.date.today()} </filecreationdate>
        <description> Generated aircraft model from simple_uav.py </description>
    </fileheader>

    <metrics>
        <wingarea unit="M2"> {params['S']} </wingarea>
        <wingspan unit="M"> {params['b']} </wingspan>
        <chord unit="M"> {params['c']} </chord>
        <location name="AERORP" unit="M"> <x> 0 </x> <y> 0 </y> <z> 0 </z> </location>
        <location name="EYEPOINT" unit="M"> <x> 0 </x> <y> 0 </y> <z> 0 </z> </location>
        <location name="VRP" unit="M"> <x> 0 </x> <y> 0 </y> <z> 0 </z> </location>
    </metrics>

    <mass_balance>
        <ixx unit="KG*M2"> {ixx} </ixx>
        <iyy unit="KG*M2"> {iyy} </iyy>
        <izz unit="KG*M2"> {izz} </izz>
        <emptywt unit="KG"> {params['mass']} </emptywt>
        <location name="CG" unit="M"> <x> 0 </x> <y> 0 </y> <z> 0 </z> </location>
    </mass_balance>

    <ground_reactions>
        <contact type="STRUCTURE" name="TAIL">
            <location unit="M"> <x> 1 </x> <y> 0 </y> <z> 0 </z> </location>
            <static_friction>  0.8 </static_friction>
            <dynamic_friction> 0.5 </dynamic_friction>
            <spring_coeff unit="N/M"> 1000 </spring_coeff>
            <damping_coeff unit="N/M/SEC"> 100 </damping_coeff>
        </contact>
    </ground_reactions>

    <propulsion/>

    <flight_control name="FCS">
        <channel name="Roll">
            <summer name="Aileron Control">
                <input> fcs/aileron-cmd-norm </input>
                <output> fcs/left-aileron-pos-rad </output>
            </summer>
        </channel>
        <channel name="Pitch">
            <summer name="Elevator Control">
                <input> fcs/elevator-cmd-norm </input>
                <output> fcs/elevator-pos-rad </output>
            </summer>
        </channel>
        <channel name="Yaw">
            <summer name="Rudder Control">
                <input> fcs/rudder-cmd-norm </input>
                <output> fcs/rudder-pos-rad </output>
            </summer>
        </channel>
    </flight_control>

    <aerodynamics>
        <axis name="LIFT">
            <function name="aero/force/Lift">
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <sum>
                        <value> {params['C_L_0']} </value>
                        <product>
                            <property>aero/alpha-rad</property>
                            <value> {params['C_L_alpha']} </value>
                        </product>
                    </sum>
                </product>
            </function>
        </axis>

        <axis name="DRAG">
            <function name="aero/force/Drag">
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <sum>
                        <value> {params['C_D_0']} </value>
                        <product>
                            <property>aero/alpha-rad</property>
                            <property>aero/alpha-rad</property>
                            <value> {params['C_D_alpha']} </value>
                        </product>
                    </sum>
                </product>
            </function>
        </axis>

        <axis name="SIDE">
            <function name="aero/force/Side">
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <sum>
                        <product>
                            <property>aero/beta-rad</property>
                            <value> {params['C_Y_beta']} </value>
                        </product>
                        <product>
                            <property>fcs/rudder-pos-rad</property>
                            <value> {params['C_Y_delta_r']} </value>
                        </product>
                    </sum>
                </product>
            </function>
        </axis>

        <axis name="ROLL">
            <function name="aero/moment/Roll">
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/bw-ft</property>
                    <sum>
                        <product>
                            <property>aero/beta-rad</property>
                            <value> {params['C_l_beta']} </value>
                        </product>
                        <product>
                            <property>velocities/p-rad_sec</property>
                            <property>metrics/bw-ft</property>
                            <value> 0.5 </value>
                            <value> {params['C_l_p']} </value>
                            <quotient> <value> 1.0 </value> <sum> <property>velocities/u-fps</property> <value> 0.001 </value> </sum> </quotient>
                        </product>
                        <product>
                            <property>velocities/r-rad_sec</property>
                            <property>metrics/bw-ft</property>
                            <value> 0.5 </value>
                            <value> {params['C_l_r']} </value>
                            <quotient> <value> 1.0 </value> <sum> <property>velocities/u-fps</property> <value> 0.001 </value> </sum> </quotient>
                        </product>
                        <product>
                            <property>fcs/left-aileron-pos-rad</property>
                            <value> {params['C_l_delta_a']} </value>
                        </product>
                        <product>
                            <property>fcs/rudder-pos-rad</property>
                            <value> {params['C_l_delta_r']} </value>
                        </product>
                    </sum>
                </product>
            </function>
        </axis>

        <axis name="PITCH">
            <function name="aero/moment/Pitch">
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/cbarw-ft</property>
                    <sum>
                        <value> {params['C_m_0']} </value>
                        <product>
                            <property>aero/alpha-rad</property>
                            <value> {params['C_m_alpha']} </value>
                        </product>
                        <product>
                            <property>velocities/q-rad_sec</property>
                            <property>metrics/cbarw-ft</property>
                            <value> 0.5 </value>
                            <value> {params['C_m_q']} </value>
                            <quotient> <value> 1.0 </value> <sum> <property>velocities/u-fps</property> <value> 0.001 </value> </sum> </quotient>
                        </product>
                        <product>
                            <property>fcs/elevator-pos-rad</property>
                            <value> {params['C_m_delta_e']} </value>
                        </product>
                    </sum>
                </product>
            </function>
        </axis>

        <axis name="YAW">
            <function name="aero/moment/Yaw">
                <product>
                    <property>aero/qbar-psf</property>
                    <property>metrics/Sw-sqft</property>
                    <property>metrics/bw-ft</property>
                    <sum>
                        <product>
                            <property>aero/beta-rad</property>
                            <value> {params['C_n_beta']} </value>
                        </product>
                        <product>
                            <property>velocities/p-rad_sec</property>
                            <property>metrics/bw-ft</property>
                            <value> 0.5 </value>
                            <value> {params['C_n_p']} </value>
                            <quotient> <value> 1.0 </value> <sum> <property>velocities/u-fps</property> <value> 0.001 </value> </sum> </quotient>
                        </product>
                        <product>
                            <property>velocities/r-rad_sec</property>
                            <property>metrics/bw-ft</property>
                            <value> 0.5 </value>
                            <value> {params['C_n_r']} </value>
                            <quotient> <value> 1.0 </value> <sum> <property>velocities/u-fps</property> <value> 0.001 </value> </sum> </quotient>
                        </product>
                        <product>
                            <property>fcs/left-aileron-pos-rad</property>
                            <value> {params['C_n_delta_a']} </value>
                        </product>
                        <product>
                            <property>fcs/rudder-pos-rad</property>
                            <value> {params['C_n_delta_r']} </value>
                        </product>
                    </sum>
                </product>
            </function>
        </axis>
    </aerodynamics>
</fdm_config>
"""
    aircraft_dir = f"src/aircraft/{aircraft_name}"
    os.makedirs(aircraft_dir, exist_ok=True)
    xml_path = f"{aircraft_dir}/{aircraft_name}.xml"
    with open(xml_path, "w") as f:
        f.write(xml_content)
    return xml_path

# ==============================================================================
# Visualization Logic (Mirroring jfsim)
# ==============================================================================
def visualize_results(history: State, time: np.ndarray, png_path: str):
    # Calculate Euler Angles for plotting
    v_quat_to_euler = jax.vmap(quat_to_euler)
    euler_angles = v_quat_to_euler(jnp.array(history.quat)) * (180.0 / jnp.pi)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'FDM: JSBsim, Episode: {EPISODE_NAME}')
    
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

def run_episode():
    # 1. Simulation settings
    T_TOTAL = SIM_PARAMS['T_total']
    DT = SIM_PARAMS['dt']
    STEPS = int(T_TOTAL / DT)
    
    # 2. Controls
    controls = create_controls(STEPS, DT)
    
    # 3. Generate JSBSim XML
    xml_path = generate_jsbsim_xml(AIRCRAFT_PARAMS)
    
    # 4. Setup JSBSim
    fdm = jsbsim.FGFDMExec("src") 
    fdm.load_model("simple_uav")
    
    # Initial Conditions (Using INIT_PARAMS)
    fdm.set_dt(DT)
    
    # Set properties for initial state
    # Position (NED)
    fdm.set_property_value("ic/pn-ft", float(INIT_PARAMS['pos'][0] * 3.28084))
    fdm.set_property_value("ic/pe-ft", float(INIT_PARAMS['pos'][1] * 3.28084))
    fdm.set_property_value("ic/h-sl-ft", float(-INIT_PARAMS['pos'][2] * 3.28084))
    
    # Velocity (Body)
    fdm.set_property_value("ic/u-fps", float(INIT_PARAMS['vel'][0] * 3.28084))
    fdm.set_property_value("ic/v-fps", float(INIT_PARAMS['vel'][1] * 3.28084))
    fdm.set_property_value("ic/w-fps", float(INIT_PARAMS['vel'][2] * 3.28084))
    
    # Orientation (Converted from Quat to Euler for JSBSim IC)
    euler_init = quat_to_euler(INIT_PARAMS['quat'])
    fdm.set_property_value("ic/phi-rad", float(euler_init[0]))
    fdm.set_property_value("ic/theta-rad", float(euler_init[1]))
    fdm.set_property_value("ic/psi-true-rad", float(euler_init[2]))
    
    # Angular Rates
    fdm.set_property_value("ic/p-rad_sec", float(INIT_PARAMS['omega'][0]))
    fdm.set_property_value("ic/q-rad_sec", float(INIT_PARAMS['omega'][1]))
    fdm.set_property_value("ic/r-rad_sec", float(INIT_PARAMS['omega'][2]))
    
    # Initialize JSBSim with these ICs
    fdm.run_ic()
    
    # 5. Simulation Loop
    print("Running JSBSim Simulation...")
    
    history_pos = []
    history_vel = []
    history_quat = []
    history_omega = []
    
    # Record initial state
    init_pos_n = float(INIT_PARAMS['pos'][0])
    init_pos_e = float(INIT_PARAMS['pos'][1])
    
    history_pos.append([
        init_pos_n + fdm.get_property_value("position/distance-from-start-lat-mt"),
        init_pos_e + fdm.get_property_value("position/distance-from-start-lon-mt"),
        -fdm.get_property_value("position/h-sl-meters")
    ])
    history_vel.append([
        fdm.get_property_value("velocities/u-fps") * 0.3048,
        fdm.get_property_value("velocities/v-fps") * 0.3048,
        fdm.get_property_value("velocities/w-fps") * 0.3048
    ])
    # Convert Euler to Quat for history
    history_quat.append(quat_from_euler(
        fdm.get_property_value("attitude/phi-rad"),
        fdm.get_property_value("attitude/theta-rad"),
        fdm.get_property_value("attitude/psi-rad")
    ))
    history_omega.append([
        fdm.get_property_value("velocities/p-rad_sec"),
        fdm.get_property_value("velocities/q-rad_sec"),
        fdm.get_property_value("velocities/r-rad_sec")
    ])
    
    for i in range(STEPS):
        # Apply controls
        fdm.set_property_value("fcs/aileron-cmd-norm", float(controls.da[i]))
        fdm.set_property_value("fcs/elevator-cmd-norm", float(controls.de[i]))
        fdm.set_property_value("fcs/rudder-cmd-norm", float(controls.dr[i]))
        fdm.set_property_value("fcs/throttle-cmd-norm", float(controls.dt[i]))
        
        # Step
        fdm.run()
        
        # Record
        history_pos.append([
            init_pos_n + fdm.get_property_value("position/distance-from-start-lat-mt"),
            init_pos_e + fdm.get_property_value("position/distance-from-start-lon-mt"),
            -fdm.get_property_value("position/h-sl-meters")
        ])
        history_vel.append([
            fdm.get_property_value("velocities/u-fps") * 0.3048,
            fdm.get_property_value("velocities/v-fps") * 0.3048,
            fdm.get_property_value("velocities/w-fps") * 0.3048
        ])
        history_quat.append(euler_to_quat(
            fdm.get_property_value("attitude/phi-rad"),
            fdm.get_property_value("attitude/theta-rad"),
            fdm.get_property_value("attitude/psi-rad")
        ))
        history_omega.append([
            fdm.get_property_value("velocities/p-rad_sec"),
            fdm.get_property_value("velocities/q-rad_sec"),
            fdm.get_property_value("velocities/r-rad_sec")
        ])
    
    print("Simulation Complete.")
    
    # Convert to matching format
    history = State(
        pos=np.array(history_pos),
        vel=np.array(history_vel),
        quat=np.array(history_quat),
        omega=np.array(history_omega)
    )
    
    time_array = np.linspace(0, T_TOTAL, STEPS + 1)
    
    return history, time_array 

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    history, time_array = run_episode()
    
    # 6. Post-Process & Save
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"jsbsim_{EPISODE_NAME}_{timestamp}"
    csv_filename = filename + ".csv"
    png_filename = filename + ".png"
    csv_path = f"logs/jsbsim/{EPISODE_NAME}/{timestamp}/{csv_filename}"
    png_path = f"logs/jsbsim/{EPISODE_NAME}/{timestamp}/{png_filename}"
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    visualize_results(history, time_array, png_path)
    
    print(f"Saving results to {csv_path}...")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["time", "pos_n", "pos_e", "pos_d", "vel_u", "vel_v", "vel_w", "quat_0", "quat_1", "quat_2", "quat_3", "omega_p", "omega_q", "omega_r"]
        writer.writerow(header)
        for i in range(len(time_array)):
            row = [time_array[i]] + list(history.pos[i]) + list(history.vel[i]) + list(history.quat[i]) + list(history.omega[i])
            writer.writerow(row)
            
    print("CSV saved successfully.")
