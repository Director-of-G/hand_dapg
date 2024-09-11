import sys
sys.path.append("/home/yongpeng/research/dexterous/test_mujoco")

import mj_envs
import click 
import os
import gym
import numpy as np
import joblib, pickle
from mjrl.utils.gym_env import GymEnv

from track_contact_algo import ContactTracker

DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

INIT_QPOS = np.array([0, -0.2, 0.5, -np.pi/2, 0, 0]+[0]*24)
INIT_QPOS[6:12] = np.array([0.0, 0.537, 0.0, 0.256, 0.288, 0.0356])

INIT_STATE = {
    "qpos": INIT_QPOS,
    "qvel": np.zeros(6+24),
}
DOF_VEL_SCALE = 0.1

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
def main(env_name):
    if env_name is "":
        print("Unknown env.")
        return
    # render demonstrations
    demo_playback(env_name)

def demo_playback(env_name):
    demo_tactile = joblib.load("/home/yongpeng/research/dexterous/hand_dapg/dapg/debug/demo_tactile.pkl")[0]
    for key in demo_tactile:
        if key != "ffdistal":
            demo_tactile[key] = np.zeros_like(demo_tactile[key])
        else:
            # demo_tactile[key] = np.ones_like(demo_tactile[key])
            demo_tactile[key] = np.array([
                [0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4]
            ])
    print("desired tactile: ", demo_tactile)

    cfg = {
        "nn": {
            "input_dim": 3,
            "output_dim": (4, 4)
        },
    }

    # initialize ContactTracker
    origin_dir = os.getcwd()
    os.chdir("/home/yongpeng/research/dexterous/test_mujoco")
    algo = ContactTracker(
        nn_dir="./model",
        xml_dir="./Adroit/Adroit_hand_kin_v2.xml",
        cfg=cfg
    )
    os.chdir(origin_dir)

    e = GymEnv(env_name)
    e.reset()
    e.set_env_state(INIT_STATE)
    if hasattr(e.env.env, "reset_object_properties"):
        e.env.env.reset_object_properties()
    actions = np.repeat(np.expand_dims(INIT_QPOS, 0), 1000, axis=0)

    tactile_data = []
    current_dof_target = INIT_QPOS[6:]

    e.env.env.sim["data"].ctrl = INIT_QPOS
    for t in range(2000):
        e.step(np.concatenate((INIT_QPOS[:6], current_dof_target)))

        env_state = e.env.env.get_env_state()
        tactile_data.append(e.env.env.taxel_data)
        print("ffdistal activation: ", e.env.env.taxel_data["ffdistal"])
        
        # compute hand dof movement
        infos = algo.get_hand_dof_movement(
            wrist_qpos=env_state['hand_qpos'][:6],
            qpos=env_state['hand_qpos'][6:],
            desired_tactile=demo_tactile,
            current_tactile=e.env.env.taxel_data
        )
        current_dof_target = current_dof_target + DOF_VEL_SCALE * infos['dof_movement']
        print("tactile loss: ", infos['tactile_loss'])

        e.env.env.render_panel_for_debug(infos['panel_fake_world_rtrans'])
        e.env.mj_render()
    breakpoint()

if __name__ == '__main__':
    main()