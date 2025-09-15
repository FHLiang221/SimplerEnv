#!/usr/bin/env python3
"""
Fix existing dataset by converting world-frame rotations to EE-frame rotations.
This script processes .npz files and updates the rotation components to match OpenVLA's expected format.
"""

import numpy as np
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation as R

def transform_rotation_to_ee_frame(world_rotation, eef_pose_vec):
    """
    Transform rotation deltas from world frame to end-effector frame.

    Args:
        world_rotation: [droll, dpitch, dyaw] in world frame
        eef_pose_vec: [x, y, z, qx, qy, qz, qw] - end-effector pose

    Returns:
        ee_rotation: [droll, dpitch, dyaw] in end-effector frame
    """
    # Extract quaternion from pose vector
    quat = eef_pose_vec[3:7]  # [qx, qy, qz, qw]

    # Convert to rotation matrix
    rotation_matrix = R.from_quat(quat).as_matrix()

    # Transform rotation deltas to EE frame
    ee_rotation = rotation_matrix @ world_rotation

    return ee_rotation

def fix_episode_file(input_file, output_file=None, backup=True):
    """
    Fix rotation frame in a single episode file.

    Args:
        input_file: Path to input .npz file
        output_file: Path to output file (if None, overwrites input)
        backup: Whether to create backup of original file
    """
    input_path = Path(input_file)

    if output_file is None:
        output_path = input_path
        backup_path = input_path.with_suffix('.npz.backup') if backup else None
    else:
        output_path = Path(output_file)
        backup_path = None

    print(f"Processing {input_path}")

    # Load data
    data = np.load(input_path)

    # Extract arrays
    actions = data['actions']  # [T, 7] - currently [EE_trans, world_rot, gripper]
    states = data['states']    # [T, 8] - [EEF pose (7D), gripper (1D)]

    T = len(actions)
    fixed_actions = actions.copy()

    # Process each timestep
    for t in range(T):
        world_rotation = actions[t, 3:6]  # [droll, dpitch, dyaw] in world frame
        eef_pose_vec = states[t, :7]      # [x, y, z, qx, qy, qz, qw]

        # Transform to EE frame
        ee_rotation = transform_rotation_to_ee_frame(world_rotation, eef_pose_vec)

        # Update action
        fixed_actions[t, 3:6] = ee_rotation

    # Create backup if requested
    if backup_path:
        print(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy2(input_path, backup_path)

    # Save fixed data
    fixed_data = {key: data[key] for key in data.files}
    fixed_data['actions'] = fixed_actions

    # Add metadata about the fix
    if 'control_mode' in fixed_data:
        fixed_data['control_mode'] = 'ee_frame_fixed'

    np.savez_compressed(output_path, **fixed_data)

    print(f"Fixed rotation frame: {output_path}")
    print(f"  Original actions shape: {actions.shape}")
    print(f"  Fixed actions shape: {fixed_actions.shape}")
    print(f"  Rotation delta change (mean): {np.mean(np.abs(fixed_actions[:, 3:6] - actions[:, 3:6])):.6f}")

def fix_dataset_directory(input_dir, output_dir=None, pattern="*.npz", backup=True):
    """
    Fix all episode files in a directory.

    Args:
        input_dir: Directory containing .npz files
        output_dir: Output directory (if None, overwrites input files)
        pattern: File pattern to match
        backup: Whether to create backups
    """
    input_path = Path(input_dir)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path

    # Find all matching files
    episode_files = list(input_path.glob(pattern))

    if not episode_files:
        print(f"No files found matching {pattern} in {input_path}")
        return

    print(f"Found {len(episode_files)} files to process")

    for episode_file in episode_files:
        if output_dir:
            output_file = output_path / episode_file.name
        else:
            output_file = None

        try:
            fix_episode_file(episode_file, output_file, backup)
        except Exception as e:
            print(f"Error processing {episode_file}: {e}")

    print(f"\n✅ Completed processing {len(episode_files)} files")
    if output_dir:
        print(f"Fixed files saved to: {output_path}")
    else:
        print("Files updated in place")
        if backup:
            print("Original files backed up with .backup extension")

def main():
    parser = argparse.ArgumentParser(description="Fix rotation frame in existing episode data")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory (default: overwrite input)")
    parser.add_argument("--pattern", default="*.npz", help="File pattern for directory processing")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # Process single file
        fix_episode_file(args.input, args.output, not args.no_backup)
    elif input_path.is_dir():
        # Process directory
        fix_dataset_directory(args.input, args.output, args.pattern, not args.no_backup)
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()

# Usage examples:
# python fix_rotation_frame.py /path/to/episode.npz
# python fix_rotation_frame.py /path/to/dataset_dir/
# python fix_rotation_frame.py /path/to/dataset_dir/ -o /path/to/fixed_dataset/
# python fix_rotation_frame.py /path/to/dataset_dir/ --pattern "*success*.npz"