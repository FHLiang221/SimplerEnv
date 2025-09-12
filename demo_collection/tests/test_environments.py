#!/usr/bin/env python3
"""
Quick test script to verify environment visual setups for debugging.
"""
import gymnasium as gym
import mani_skill2_real2sim.envs
from mani_skill2_real2sim.utils.sapien_utils import look_at
import numpy as np
from sapien.core import Pose

# Test environments
test_envs = [
    ("GraspSingleCokeCanInScene-v0", "Coke Can (should have black table + gray walls)"),
    ("GraspSingleRandomObjectInScene-v0", "Random Object (should have black table + gray walls)"),
    ("MoveNearGoogleBakedTexInScene-v1", "Move Near (should have black table + gray walls)")
]

def test_environment(env_id, description):
    print(f"\n=== Testing {env_id} ===")
    print(f"Description: {description}")
    
    try:
        # Camera setup similar to rlds_jaco.py
        pose = look_at([1.0, 1.0, 2.5], [0.0, 0.0, 0.7])
        camera_cfgs = {
            "base_camera": dict(p=pose.p, q=pose.q, width=128, height=128, fov=np.deg2rad(69.4))
        }
        
        env_kwargs = {
            "obs_mode": "rgbd",
            "control_mode": "arm_pd_ee_target_delta_pose_gripper_pd_joint_pos",
            "robot": "jaco",
            "sim_freq": 501,
            "control_freq": 3,
            "camera_cfgs": camera_cfgs,
        }
        
        # Don't specify scene_name - let environment use its defaults
        env = gym.make(env_id, **env_kwargs)
        
        # Test reset options for move near
        reset_options = {
            "robot_init_options": {
                "init_xy": [-0.45, 0.6],
                "init_rot_quat": Pose(q=[1, 0, 0, 0]).q,
            }
        }
        
        if "MoveNear" in env_id:
            reset_options["obj_init_options"] = {"episode_id": 0}
        
        # Reset with fixed seed for consistency
        obs, info = env.reset(seed=42, options=reset_options)
        
        print(f"✅ {env_id} created successfully")
        print(f"   Robot position: {env.agent.robot.pose.p}")
        print(f"   Scene actors: {len(env._scene.get_all_actors())} total")
        
        # Check for visual elements by looking at actors
        actors = env._scene.get_all_actors()
        visual_actors = [actor for actor in actors if len(actor.get_visual_bodies()) > 0]
        print(f"   Visual actors: {len(visual_actors)}")
        
        # Look for dummy_table actors (black table/gray walls indicators)
        dummy_tables = [actor for actor in actors if "dummy_table" in str(actor) or actor.name == ""]
        print(f"   Potential visual elements (unnamed actors): {len(dummy_tables)}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Error testing {env_id}: {e}")
        return False

if __name__ == "__main__":
    print("Testing environment visual setups...")
    
    success_count = 0
    for env_id, description in test_envs:
        if test_environment(env_id, description):
            success_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully tested {success_count}/{len(test_envs)} environments")
    
    if success_count == len(test_envs):
        print("✅ All environments should now have proper visual setups!")
        print("Try running:")
        print("  python rlds_jaco.py google_robot_pick_object 1")  
        print("  python rlds_jaco.py google_robot_move_near 1")
    else:
        print("❌ Some environments had issues - check the errors above")