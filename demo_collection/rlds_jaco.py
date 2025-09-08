#!/usr/bin/env python3
"""
Collect tele-operated trajectories in SimplerEnv with a Nintendo Switch
controller and save to RLDS format for OpenVLA training - Jaco Arm version.

This script follows the kpertsch/rlds_dataset_builder pattern for proper OpenVLA compatibility.
Modified specifically for Kinova Jaco arm with appropriate action/proprio dimensions.
MODIFIED: Only saves successful episodes as per PI guidance.
"""

from datetime import datetime
from pathlib import Path
import time
import socket
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import fire
import simpler_env
from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict,
)
from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose

# ── Jaco Arm Constants ──────────────────────────────────────────────────────
# Kinova Jaco arm - using 8D proprioception for OpenVLA-OFT compatibility
JACO_ACTION_DIM = 7  # 6D end-effector pose delta + 1D gripper
JACO_PROPRIO_DIM = 8  # EEF pose (7D) + gripper (1D) - matches LIBERO format
JACO_NUM_ACTIONS_CHUNK = 8  # Similar to LIBERO for consistency

# ── TCP receiver (connect-once, non-blocking reads) ────────────────────
TCP_PORT = 5555
_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
_sock.bind(("0.0.0.0", TCP_PORT))
_sock.listen(1)
print("Waiting for controller stream on TCP 5555 …")
_conn, _a = _sock.accept()                 # blocks until the Mac connects
_conn.setblocking(False)

_last = np.zeros(10, dtype=np.float32)    # [action(7), plus flag, start flag, minus flag]

# ── Episode-level language helper ──────────────────────────────────────
def episode_instruction(env, info, env_name: str) -> str:
    if "target_drawer" in info:                     # any drawer task
        drawer = info["target_drawer"]
        obj = info.get("object_name")

        # recognise **all** place-drawer envs, even custom names
        if obj is None and "place" in env_name.lower():
            if getattr(env, "model_id", None):
                obj = env._get_instruction_obj_name(env.model_id)

        if obj:                                     # place-drawer
            return f"place {obj} into {drawer} drawer"

        # open-drawer / close-drawer
        return env.get_language_instruction()

    # pick / move-near / …
    return env.get_language_instruction()

def get_switch_action() -> np.ndarray:
    """Return the latest 7-float action vector (dx … gripper) for Jaco arm."""
    global _last
    try:
        data = _conn.recv(40)             # exactly 10 floats = 40 bytes
        if data:                          # ignore empty packets
            _last = np.frombuffer(data, dtype=np.float32).copy()  # Make writable copy
    except BlockingIOError:
        pass                              # nothing arrived this frame
    return _last[:JACO_ACTION_DIM]        # first 7 floats = action for Jaco

def wants_quit() -> bool:
    """PLUS button flag sent as the 8th float."""
    return bool(_last[7])

def wants_start() -> bool:
    """PLUS button flag sent as the 8th float (same as quit)."""
    return bool(_last[7])

def wants_discard() -> bool:
    """MINUS button flag sent as the 10th float."""
    return bool(_last[9])

def clear_start_flag():
    """Clear the start flag after episode begins."""
    global _last
    _last[7] = 0.0

def clear_discard_flag():
    """Clear the discard flag after handling."""
    global _last
    _last[9] = 0.0

def clear_all_flags():
    """Clear all button flags."""
    global _last
    _last[7] = 0.0  # plus/quit/start
    _last[8] = 0.0  # home (unused)
    _last[9] = 0.0  # minus/discard

def prompt_failure_decision(episode_data, step_count):
    """Prompt user to decide whether to save or discard a failed episode."""
    instruction = episode_data['language_instruction']
    
    print(f"\n🤔 FAILURE DECISION:")
    print(f"   Task: {instruction}")
    print(f"   Steps: {step_count}")
    print(f"   + button = SAVE failure (worth analyzing)")
    print(f"   - button = DISCARD failure (not useful)")
    print("   Choose now...")
    
    # Clear any existing button presses
    clear_all_flags()
    
    # Wait for decision
    while True:
        get_switch_action()  # Update button states
        
        if wants_quit():  # + button = save failure
            clear_all_flags()
            print("💾 Saving failure for analysis")
            return True
        elif wants_discard():  # - button = discard failure  
            clear_all_flags()
            print("🗑️  Discarding failure")
            return False
            
        time.sleep(0.1)

# ── matplotlib defaults (prevent key clashes) ──────────────────────────
plt.rcParams["keymap.quit"].remove("q")
plt.rcParams["keymap.save"].remove("s")

# ── Jaco-specific proprioception extraction ────────────────────────────
def get_jaco_proprioception(env, obs) -> np.ndarray:
    """
    Extract Jaco arm proprioceptive state compatible with OpenVLA-OFT.
    Returns 8D vector: [EEF pose (7D: x,y,z,qx,qy,qz,qw), gripper (1D)]
    Same format as LIBERO for compatibility.
    """
    # Get end-effector pose relative to robot base (more stable for mobile robots)
    eef_pose_world = env.tcp.pose
    eef_pose_relative = env.agent.robot.pose.inv() * eef_pose_world
    eef_pose_vec = vectorize_pose(eef_pose_relative)  # [x,y,z,qx,qy,qz,qw]
    
    # Get gripper state from proprioception
    agent_proprio = obs["agent"]
    qpos = agent_proprio["qpos"]
    # For Jaco, gripper joints are typically the last joints
    # We'll use the first gripper joint as the gripper openness indicator
    gripper_state = float(qpos[-1]) if len(qpos) > 7 else 0.0
    
    # Build 8D state vector: EEF pose (7D) + gripper (1D)
    return np.concatenate([eef_pose_vec, [gripper_state]], axis=0).astype(np.float32)

# ── Episode Data Collection and Storage ─────────────────────────────────
def collect_episode_data(env, env_name: str, episode_id: int, ignore_quit_for: int):
    """Collect a single episode and return the raw data with success status."""
    
    obs, info = env.reset()
    instruction = episode_instruction(env, info, env_name)   # ← pass env_name

    
    # Show the task instruction prominently
    print(f"\n📋 TASK FOR THIS EPISODE:")
    print(f"   {instruction}")
    print("=" * (len(instruction) + 6))
    print("🎮 Start demonstrating with Jaco arm! (+ or - button to abort episode if needed)\n")
    
    episode_data = {
        'episode_id': episode_id,
        'language_instruction': instruction,  # ← Now episode-specific
        'environment_name': env_name,
        'robot_type': 'jaco',  # ← Explicitly mark as Jaco data
        'steps': []
    }

    step_count = 0
    
    while True:
        # Show RGB observation
        img = Image.fromarray(get_image_from_maniskill2_obs_dict(env, obs))
        resized = np.array(img.resize((224, 224), Image.BILINEAR))
        plt.imshow(img)
        plt.axis("off")
        plt.draw()
        plt.pause(0.001)
        plt.cla()

        # Check for manual episode termination
        if ignore_quit_for > 0:          # still in debounce window
            ignore_quit_for -= 1
        elif wants_quit() or wants_discard():  # + or - button pressed
            if wants_quit():
                print(f"\nEpisode aborted by + button (step {step_count})")
            else:
                print(f"\nEpisode aborted by - button (step {step_count})")
            
            episode_data['success'] = False
            episode_data['manual_abort'] = True
            clear_all_flags()
            return episode_data

        action = get_switch_action()
        
        print(
            f"\rStep {step_count:3d} | Jaco Action: "
            + " | ".join(f"{a:+.02f}" for a in action),
            end="",
            flush=True,
        )

        # Step only if something changed
        if np.any(np.abs(action) > 1e-3):
            obs, reward, success, truncated, _ = env.step(action)
            done = success or truncated
            step_count += 1

            # --- Build Jaco-specific proprioceptive state ---
            jaco_proprio = get_jaco_proprioception(env, obs)  # 8D: EEF pose + gripper
            
            # Store step data with episode-specific instruction
            step_data = {
                'image': resized.astype(np.uint8),
                'state': jaco_proprio.tolist(),  # ← Jaco-specific 8D proprio vector (EEF+gripper)
                'language_instruction': instruction,  # ← Episode-specific instruction
                'action': action.tolist(),
                'reward': float(reward),
                'is_terminal': bool(done),
                'is_first': (step_count == 1),
            }
            
            episode_data['steps'].append(step_data)

            if done:
                if success:
                    print(f"\n🎉 Jaco episode completed successfully! (step {step_count})")
                    print(f"   Task: {instruction}")  # ← Show what task was completed
                    print(f"   Final reward: {reward:.3f}")
                    episode_data['success'] = True
                    return episode_data
                else:
                    print(f"\n❌ Jaco episode truncated/failed (step {step_count})")
                    print(f"   Task was: {instruction}")
                    print(f"   Final reward: {reward:.3f}")
                    episode_data['success'] = False
                    episode_data['natural_failure'] = True
                    return episode_data
        else:
            time.sleep(0.01)

    return None

def save_episode_to_file(episode_data, success_dir: Path, failure_dir: Path):
    """Save episode data to appropriate folder based on success status."""
    is_success = episode_data.get('success', False)
    output_dir = success_dir if is_success else failure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    status_prefix = "success" if is_success else "failure"
    episode_file = output_dir / f"jaco_{status_prefix}_episode_{episode_data['episode_id']:06d}.npz"
    
    steps = episode_data["steps"]
    T = len(steps)
    if T == 0:
        raise ValueError("Empty episode; nothing to save.")
    
    # Stack arrays per field (no pickled lists)
    images = np.stack([s["image"] for s in steps], axis=0)                    # [T, 224, 224, 3] uint8
    actions = np.stack([s["action"] for s in steps], axis=0).astype(np.float32) # [T, 7]
    states = np.stack([s["state"] for s in steps], axis=0).astype(np.float32)  # [T, 8] (eef_pose+gripper)
    rewards = np.asarray([s["reward"] for s in steps], dtype=np.float32)       # [T]
    is_terminal = np.asarray([s["is_terminal"] for s in steps], dtype=bool)    # [T]
    is_first = np.asarray([s["is_first"] for s in steps], dtype=bool)          # [T]
    
    # Debug: print shapes to verify correctness for Jaco
    print(f"Jaco Shapes: images={images.shape}, actions={actions.shape}, states={states.shape}, "
          f"rewards={rewards.shape}, is_terminal={is_terminal.shape}, is_first={is_first.shape}")
    
    np.savez_compressed(
        episode_file,
        episode_id=np.int64(episode_data["episode_id"]),
        env_name=episode_data["environment_name"],
        robot_type=episode_data["robot_type"],
        language=episode_data["language_instruction"],
        images=images,
        actions=actions,
        states=states,
        rewards=rewards,
        is_terminal=is_terminal,
        is_first=is_first,
    )
    status_emoji = "✅" if is_success else "❌"
    status_text = "SUCCESS" if is_success else "FAILURE"
    print(f"💾 {status_emoji} JACO {status_text}: Episode {episode_data['episode_id']} saved to {episode_file}")
    return episode_file

# ── MAIN TRAJECTORY COLLECTION LOGIC ───────────────────────────────────
def collect_trajectory(env_name: str, num_trajs: int):
    """
    Collect `num_trajs` Switch-tele-operated demos in SimplerEnv `env_name` for Jaco arm.
    Saves episodes as individual files for later RLDS conversion.
    ONLY SAVES SUCCESSFUL EPISODES.
    """
    
    # Create output directories with timestamp - save to SimplerEnv/demo_collection/jaco_data
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset_name = f"jaco_{env_name}_{num_trajs}trajs_switch_{timestamp}_all_episodes"
    jaco_data_dir = Path("/project/fhliang/projects/SimplerEnv/demo_collection/jaco_data")
    jaco_data_dir.mkdir(exist_ok=True)  # Create jaco_data dir if it doesn't exist
    base_dir = jaco_data_dir / dataset_name
    success_dir = base_dir / "successes"
    failure_dir = base_dir / "failures"
    
    plt.figure()
    
    # Create environment with Jaco robot configuration - use gymnasium directly
    # Map common task names to ManiSkill2 environment IDs and their parameters
    env_mapping = {
        "google_robot_pick_coke_can": ("GraspSingleOpenedCokeCanInScene-v0", {}),
        "google_robot_pick_horizontal_coke_can": ("GraspSingleOpenedCokeCanInScene-v0", {"lr_switch": True}),
        "google_robot_pick_vertical_coke_can": ("GraspSingleOpenedCokeCanInScene-v0", {"laid_vertically": True}),
        "google_robot_pick_standing_coke_can": ("GraspSingleOpenedCokeCanInScene-v0", {"upright": True}),
        "google_robot_pick_object": ("GraspSingleRandomObjectInScene-v0", {}),
        "google_robot_move_near": ("MoveNearGoogleBakedTexInScene-v1", {}),
        "google_robot_move_near_v0": ("MoveNearGoogleBakedTexInScene-v0", {}),
        "google_robot_move_near_v1": ("MoveNearGoogleBakedTexInScene-v1", {}),
        "google_robot_open_drawer": ("OpenDrawerCustomInScene-v0", {}),
        "google_robot_open_top_drawer": ("OpenTopDrawerCustomInScene-v0", {}),
        "google_robot_open_middle_drawer": ("OpenMiddleDrawerCustomInScene-v0", {}),
        "google_robot_open_bottom_drawer": ("OpenBottomDrawerCustomInScene-v0", {}),
        "google_robot_close_drawer": ("CloseDrawerCustomInScene-v0", {}),
        "google_robot_close_top_drawer": ("CloseTopDrawerCustomInScene-v0", {}),
        "google_robot_close_middle_drawer": ("CloseMiddleDrawerCustomInScene-v0", {}),
        "google_robot_close_bottom_drawer": ("CloseBottomDrawerCustomInScene-v0", {}),
        "google_robot_place_in_closed_drawer": ("PlaceIntoClosedDrawerCustomInScene-v0", {}),
        "google_robot_place_in_closed_top_drawer": ("PlaceIntoClosedTopDrawerCustomInScene-v0", {}),
        "google_robot_place_in_closed_middle_drawer": ("PlaceIntoClosedMiddleDrawerCustomInScene-v0", {}),
        "google_robot_place_in_closed_bottom_drawer": ("PlaceIntoClosedBottomDrawerCustomInScene-v0", {}),
        "google_robot_place_apple_in_closed_top_drawer": ("PlaceIntoClosedTopDrawerCustomInScene-v0", {"model_ids": "baked_apple_v2"}),
        "widowx_spoon_on_towel": ("PutSpoonOnTableClothInScene-v0", {}),
        "widowx_carrot_on_plate": ("PutCarrotOnPlateInScene-v0", {}),
        "widowx_stack_cube": ("StackGreenCubeOnYellowCubeBakedTexInScene-v0", {}),
        "widowx_put_eggplant_in_basket": ("PutEggplantInBasketScene-v0", {}),
    }
    
    # Get the actual ManiSkill2 environment ID and parameters
    if env_name in env_mapping:
        maniskill_env_id, task_params = env_mapping[env_name]
    else:
        # Assume it's already a ManiSkill2 environment ID
        maniskill_env_id, task_params = env_name, {}
    
    import gymnasium as gym
    import mani_skill2_real2sim.envs  # Register the environments
    
    # Combine task-specific parameters with Jaco robot configuration
    env_kwargs = {
        "obs_mode": "rgbd",
        "control_mode": "arm_pd_ee_target_delta_pose_gripper_pd_joint_pos",
        "robot": "jaco",
        "sim_freq": 501,
        "control_freq": 3,
        "scene_name": "google_pick_coke_can_1_v4",
        **task_params  # Add task-specific parameters (lr_switch, upright, etc.)
    }
    
    env = gym.make(maniskill_env_id, **env_kwargs)

    print(f"Collecting Jaco arm trajectories in {env_name}")
    print(f"Saving to: {base_dir}")
    print("💾 ALL EPISODES MODE: Both successes and failures will be saved")
    print(f"  Successes → {success_dir}")
    print(f"  Failures → {failure_dir}")
    print(f"🤖 JACO ARM MODE: Action dim={JACO_ACTION_DIM}, Proprio dim={JACO_PROPRIO_DIM}")
    print("📝 ADAPTIVE MODE: Instructions captured per episode for randomized tasks")
    print(
        "Controls: Left hand (L-stick, L/ZL) = translation, "
        "Right hand (R-stick, R/ZR) = rotation, A/B = gripper"
    )
    print("Episode control: + = start episode / abort episode, - = abort episode")
    print("After failure: + = save failure, - = discard failure")

    collected_successes = []
    collected_failures = []
    discarded_failures = 0
    attempted_episodes = 0
    unique_instructions = set()

    ep_i = 1
    while len(collected_successes) < num_trajs:
        attempted_episodes += 1
        print(f"\nJaco Attempt {attempted_episodes} (Target: {len(collected_successes) + 1}/{num_trajs} successes) – press + button to start…")
        
        # Wait for + button to be released first
        while wants_start():
            time.sleep(0.1)
            get_switch_action()
        
        # Now wait for + button press to start episode
        while not wants_start():
            time.sleep(0.1)
            get_switch_action()
        
        print("Jaco episode starting...")
        clear_start_flag()
        
        ignore_quit_for = 10                # ≈ 10 * 0.03 s  ➜  0.3 s debounce
        episode_data = collect_episode_data(env, env_name, ep_i, ignore_quit_for)
        
        if episode_data:
            unique_instructions.add(episode_data['language_instruction'])
            
            if episode_data.get('success', False):
                # Success - always save
                episode_file = save_episode_to_file(episode_data, success_dir, failure_dir)
                collected_successes.append(episode_file)
                print(f"✅ Jaco Success {len(collected_successes)}/{num_trajs} collected")
            else:
                # Failure - check if episode has any steps
                step_count = len(episode_data.get('steps', []))
                
                if step_count == 0:
                    # Empty episode - automatically discard
                    discarded_failures += 1
                    print(f"🗑️  Empty Jaco episode discarded automatically (still need {num_trajs - len(collected_successes)} successes)")
                else:
                    # Non-empty failure - prompt for decision
                    should_save = prompt_failure_decision(episode_data, step_count)
                    
                    if should_save:
                        episode_file = save_episode_to_file(episode_data, success_dir, failure_dir)
                        collected_failures.append(episode_file)
                        print(f"💾 Jaco Failure {len(collected_failures)} saved (still need {num_trajs - len(collected_successes)} successes)")
                    else:
                        discarded_failures += 1
                        print(f"🗑️  Jaco Failure {discarded_failures} discarded (still need {num_trajs - len(collected_successes)} successes)")
            
            ep_i += 1
        else:
            print("⚠️  Unexpected error with Jaco - trying again")

    # Save metadata with instruction diversity info
    metadata = {
        'dataset_name': dataset_name,
        'environment_name': env_name,
        'robot_type': 'jaco',
        'action_dim': JACO_ACTION_DIM,
        'proprio_dim': JACO_PROPRIO_DIM,
        'num_actions_chunk': JACO_NUM_ACTIONS_CHUNK,
        'num_successful_episodes': len(collected_successes),
        'num_saved_failures': len(collected_failures),
        'num_discarded_failures': discarded_failures,
        'total_saved_episodes': len(collected_successes) + len(collected_failures),
        'total_attempted_episodes': attempted_episodes,
        'attempted_episodes': attempted_episodes,
        'success_rate': len(collected_successes) / attempted_episodes,
        'failure_save_rate': len(collected_failures) / (len(collected_failures) + discarded_failures) if (len(collected_failures) + discarded_failures) > 0 else 0,
        'unique_instructions': list(unique_instructions),
        'instruction_diversity': len(unique_instructions),
        'success_episode_files': [str(f) for f in collected_successes],
        'failure_episode_files': [str(f) for f in collected_failures],
        'timestamp': timestamp,
        'collection_mode': 'all_episodes_adaptive_instructions',
        'compatibility': {
            'rlds_dataset_builder': True,
            'openvla_oft': True,
            'format_version': '1.0.0'
        }
    }
    
    metadata_file = base_dir / "jaco_metadata.json"
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    total_failures = len(collected_failures) + discarded_failures
    failure_save_rate = len(collected_failures) / total_failures if total_failures > 0 else 0
    
    print(f"""
✅ Jaco Collection complete! Episodes saved with quality filtering.

Data location: {base_dir}
Robot type: Jaco Arm ({JACO_ACTION_DIM}D actions, {JACO_PROPRIO_DIM}D proprioception - OpenVLA compatible)
Successful episodes: {len(collected_successes)}
Saved failures: {len(collected_failures)}
Discarded failures: {discarded_failures}
Total saved episodes: {len(collected_successes) + len(collected_failures)}
Total attempts: {attempted_episodes}
Success rate: {len(collected_successes)/attempted_episodes:.1%}
Failure save rate: {failure_save_rate:.1%}
Unique instructions: {len(unique_instructions)}

📁 Directory structure:
  📂 {success_dir}/ ({len(collected_successes)} files)
  📂 {failure_dir}/ ({len(collected_failures)} files)
  🗑️  Discarded: {discarded_failures} low-quality failures

📝 Instructions captured:
{chr(10).join(f"  • {instr}" for instr in sorted(unique_instructions))}

🔄 Next steps for Jaco data:
1. Use SUCCESS episodes for RLDS dataset builder training data
2. Analyze SAVED FAILURE episodes to understand common failure modes
3. Copy success data to rlds_dataset_builder folder
4. Create jaco-specific dataset builder (see rlds_dataset_builder/ examples)
5. Run 'tfds build' to create the RLDS dataset
6. Add Jaco constants to openvla-oft/prismatic/vla/constants.py:
   JACO_CONSTANTS = {{
       "NUM_ACTIONS_CHUNK": {JACO_NUM_ACTIONS_CHUNK},
       "ACTION_DIM": {JACO_ACTION_DIM},
       "PROPRIO_DIM": {JACO_PROPRIO_DIM},
       "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
   }}
7. Train OpenVLA with Jaco-specific configurations

💡 Pro tip: Quality filtering helps focus analysis on meaningful failures
🤖 Jaco-specific: Data format optimized for OpenVLA-OFT compatibility
    """)

# ── entry-point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    fire.Fire(collect_trajectory)
    
# Usage examples:
# python rlds_jaco.py google_robot_pick_standing_coke_can 3
# python rlds_jaco.py google_robot_close_top_drawer 5
# Now saves both successes and failures for Jaco arm analysis!