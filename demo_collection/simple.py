from datetime import datetime

import fire
import matplotlib.pyplot as plt
import simpler_env
from PIL import Image
from simpler_env.utils.env.observation_utils import \
    get_image_from_maniskill2_obs_dict

plt.rcParams['keymap.quit'].remove('q')
plt.rcParams['keymap.save'].remove('s')
import json
import time
from pathlib import Path

import numpy as np
from pynput import keyboard

# The interval to query the keyboard for actions
ACTION_INTERVAL = 0.1

# Action deltas
VEL_TRANSLATE = 0.05
VEL_ROTATE = 0.1

pressed_keys = set()
# Key press/release handlers
def on_press(key):
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        if key == keyboard.Key.shift:
            pressed_keys.add('shift')
        elif key == keyboard.Key.ctrl:
            pressed_keys.add('ctrl')
        elif key == keyboard.Key.esc:
            pressed_keys.add('esc')

def on_release(key):
    try:
        pressed_keys.discard(key.char)
    except AttributeError:
        if key == keyboard.Key.shift:
            pressed_keys.discard('shift')
        elif key == keyboard.Key.ctrl:
            pressed_keys.discard('ctrl')
        elif key == keyboard.Key.esc:
            pressed_keys.discard('esc')

# Start the key listener in the background
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def get_keyboard_action():
    """Listens to keyboard input and returns the corresponding action vector.
    
    Returns:
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    action = np.zeros(7)

    # Translation
    if 'w' in pressed_keys: action[0] = VEL_TRANSLATE  # +x
    if 's' in pressed_keys: action[0] = -VEL_TRANSLATE  # -x
    if 'a' in pressed_keys: action[1] = VEL_TRANSLATE  # -y
    if 'd' in pressed_keys: action[1] = -VEL_TRANSLATE  # +y
    if 'shift' in pressed_keys: action[2] = VEL_TRANSLATE  # +z
    if 'ctrl' in pressed_keys: action[2] = -VEL_TRANSLATE  # -z

    # Gripper
    if 'q' in pressed_keys: action[6] = -1           # open
    if 'e' in pressed_keys: action[6] = 1            # close

    # Rotation
    if 'i' in pressed_keys: action[3] = VEL_ROTATE  # +x
    if 'k' in pressed_keys: action[3] = -VEL_ROTATE  # -x
    if 'j' in pressed_keys: action[4] = VEL_ROTATE  # -y
    if 'l' in pressed_keys: action[4] = -VEL_ROTATE  # +y
    if 'u' in pressed_keys: action[5] = VEL_ROTATE  # +z
    if 'o' in pressed_keys: action[5] = -VEL_ROTATE  # -z

    return action


def collect_trajectory(env_name, num_trajs):
    """Given a SimplerEnv google robot environment name (see 
    <https://github.com/simpler-env/SimplerEnv/blob/main/simpler_env/__init__.py#L5>
    for a list of available environments), collect `num_trajs` teleoperating 
    trajectories and save them in a format compatible with OpenVLA finetuning.

    NOTE: We only step the environment upon user input to give user more time 
    to prepare. This means all collected trajectories will have the same length 
    equal to max allowed steps. Trajectories may need to be downsampled and
    filtered later to match the model's training frequency.

    Args:
        env_name (str): Name of the SimplerEnv environment to collect 
            trajectory on.
        num_trajs (int): Number of trajectories to collect.
    """
    out_dir = Path(f"./{env_name}_{num_trajs}trajs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
    out_dir.mkdir(exist_ok=True)

    plt.figure()
    env = simpler_env.make(env_name)
    instruction = env.get_language_instruction()

    print(f"Collecting trajectory for environment {env_name}")
    print(f"Prompt: {instruction}")
    print("WASD+LShift+LCtrl = Move, IJKLUO = Rotate, Q/E = Gripper, esc = quit")
    
    for i in range(num_trajs):
        print(f"Collecting trajectory {i+1}|{num_trajs}")

        episode = {
            "steps": [],
            "episode_metadata": {
                "language_instruction": instruction,
                "episode_idx": i+1,
                "environment": env_name
            }
        }

        obs, _ = env.reset()
        # TODO
        done = False
        while True:
            # google robot only has overhead and base cameras (no wrist)
            image = Image.fromarray(get_image_from_maniskill2_obs_dict(env, obs))
            resized_image = np.array(image.resize((224, 224), Image.BILINEAR))

            plt.imshow(image)
            plt.draw()
            plt.pause(0.01)
            plt.cla()

            if 'esc' in pressed_keys:
                print("\nTeleop interrupted\n")
                break
            
            action = get_keyboard_action()
            formatted_action = [f"{a:6.2f}" for a in action]
            print(f"\rAction: {' | '.join(formatted_action)}", end='', flush=True)

            if np.all(action == 0):
                time.sleep(ACTION_INTERVAL)
                continue

            obs, reward, success, truncated, _ = env.step(action)
            done = success or truncated

            step = {
                "observation": {
                    "image": resized_image.astype(np.uint8).tolist(),
                    "robot_state": {
                        "pose": obs['agent']['base_pose'].tolist()
                    },
                    "language_instruction": instruction
                },
                "action": action.tolist(),
                "reward": reward,
                "is_terminal": done,
            }
            episode["steps"].append(step)

        with open(out_dir / f"{i+1}.json", "w") as f:
            json.dump(episode, f)

if __name__ == "__main__":
    fire.Fire(collect_trajectory)
    
    # python simplerenv_env_vis.py --env_name google_robot_pick_standing_coke_can --num_trajs 3