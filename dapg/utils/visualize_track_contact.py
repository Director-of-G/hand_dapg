import os
import sys
sys.path.append("/home/yongpeng/research/dexterous/test_mujoco")
sys.path.append(os.path.join(os.path.dirname(__file__), '../debug'))

import mj_envs
import click 
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv

from plot import plot_tactile_readings
from track_contact_algo import ContactTracker

DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
def main(env_name):
    if env_name is "":
        print("Unknown env.")
        return
    demos = pickle.load(open('./demonstrations/'+env_name+'_demos.pickle', 'rb'))
    # render demonstrations
    demo_playback(env_name, demos)

def demo_playback(env_name, demo_paths):
    # temp import
    import torch
    from scipy.spatial.transform import Rotation as SciR

    def pos_euler_to_rtrans(pos, euler):
        rtrans = torch.eye(4)
        rtrans[:3, 3] = torch.from_numpy(pos)
        rtrans[:3, :3] = torch.from_numpy(SciR.from_euler("XYZ", euler).as_matrix())
        return rtrans

    cfg = {
        "nn": {
            "input_dim": 3,
            "output_dim": (4, 4)
        },
    }

    origin_dir = os.getcwd()
    os.chdir("/home/yongpeng/research/dexterous/test_mujoco")
    algo = ContactTracker(
        nn_dir="./model",
        xml_dir="./Adroit/Adroit_hand_kin_v2.xml",
        cfg=cfg
    )
    os.chdir(origin_dir)

    # read desired tactile data
    import joblib
    demo_tactile = joblib.load("/home/yongpeng/research/dexterous/hand_dapg/dapg/debug/demo_tactile.pkl")

    e = GymEnv(env_name)
    e.reset()

    # visualize a single frame
    t_inspect = 230
    demo_states = pickle.load(open('/home/yongpeng/research/dexterous/hand_dapg/dapg/debug/demo_states.pkl', 'rb'))
    path = demo_paths[0]
    states_t = demo_states[t_inspect]
    selected_jpos_ids = e.env.env.non_taxel_jpos_ids
    selected_jvel_ids = e.env.env.non_taxel_jvel_ids
    # print("obj qpos: ", states_t['qpos'][selected_jpos_ids][-6:])
    states_t['qpos'] = states_t['qpos'][selected_jpos_ids]
    states_t['qvel'] = states_t['qvel'][selected_jvel_ids]
    # states_t['qpos'][-6:] = 0.0
    states_t['obj_pos'][:] = path['init_state_dict']['obj_pos']
    e.set_env_state(states_t)

    tactile_data = demo_tactile[t_inspect]

    panel_ref_rtrans = algo.get_hand_dof_movement(
        wrist_qpos=states_t["hand_qpos"][:6],
        qpos=states_t["hand_qpos"][6:],
        desired_tactile=tactile_data,
        current_tactile=tactile_data
    )['panel_fake_world_rtrans']
    
    e.env.env.render_panel_for_debug(panel_ref_rtrans)
    e.env.mj_render()
    plot_tactile_readings(tactile_data)

    breakpoint()

    return

    # visualize the full trajectory
    for path in demo_paths:
        extras = []
        state_path = [path['init_state_dict']]

        e.set_env_state(path['init_state_dict'])
        if hasattr(e.env.env, "reset_object_properties"):
            e.env.env.reset_object_properties()
        actions = path['actions']
        for t in range(actions.shape[0]):
            e.step(actions[t])
            extras.append(e.env.env.taxel_data)
            state_path.append(e.env.env.get_env_state())

            # # debug contact tracking
            # env_state = e.env.env.get_env_state()

            # forearm_rtrans = pos_euler_to_rtrans(pos=np.array([0, -0.7, 0.2]), euler=np.array([-1.57, 0, 3.14]))
            # panel_ref_rtrans = algo.get_hand_dof_movement(
            #     qpos=env_state["hand_qpos"][6:],
            #     desired_tactile=demo_tactile[t],
            #     current_tactile=e.env.env.taxel_data
            # )['panel_fake_world_rtrans']
            # hand_wrist_rtrans = pos_euler_to_rtrans(pos=env_state["hand_qpos"][:3], euler=env_state["hand_qpos"][3:6])

            # delta_rtrans = torch.matmul(forearm_rtrans, torch.matmul(hand_wrist_rtrans, torch.linalg.inv(forearm_rtrans)))
            # panel_ref_rtrans = {k: torch.matmul(delta_rtrans, v) for k, v in panel_ref_rtrans.items()}
            # breakpoint()
            # e.env.env.render_panel_for_debug(panel_ref_rtrans)

            e.env.mj_render()
        breakpoint()

if __name__ == '__main__':
    main()
