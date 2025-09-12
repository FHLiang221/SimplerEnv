#!/usr/bin/env python3
"""
Minimal test to check if environment stepping works without segfault
"""
import simpler_env
import numpy as np
import os

# Set environment variables
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['SAPIEN_RENDERER'] = 'cpu'
if 'VK_ICD_FILENAMES' in os.environ:
    del os.environ['VK_ICD_FILENAMES']
os.environ['MS2_REAL2SIM_ASSET_DIR'] = '/project/fhliang/projects/SimplerEnv/ManiSkill2_real2sim/data'

def test_env_step():
    """Test basic environment functionality"""
    print("Creating environment...")
    env = simpler_env.make('google_robot_pick_horizontal_coke_can')
    
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Reset successful. Observation keys: {list(obs.keys())}")
    
    print("Testing environment step...")
    # Create a small random action
    action = np.array([0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    print(f"Action: {action}")
    
    try:
        obs, reward, success, truncated, info = env.step(action)
        print(f"Step successful! Reward: {reward}, Success: {success}, Truncated: {truncated}")
        return True
    except Exception as e:
        print(f"Step failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()

if __name__ == "__main__":
    success = test_env_step()
    if success:
        print("✅ Environment step test PASSED")
    else:
        print("❌ Environment step test FAILED")