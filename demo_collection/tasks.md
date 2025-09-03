# SimplerEnv Data Collection Commands

## **Picking Tasks** (50 episodes total)

```bash
python rlds.py --env_name google_robot_pick_horizontal_coke_can --num_trajs 10
python rlds.py --env_name google_robot_pick_vertical_coke_can --num_trajs 10
python rlds.py --env_name google_robot_pick_standing_coke_can --num_trajs 10
python rlds.py --env_name google_robot_pick_coke_can --num_trajs 10
python rlds.py --env_name google_robot_pick_object --num_trajs 20
```

## **Drawer Tasks**

### Opening drawers (different levels)
```bash
python rlds.py --env_name google_robot_open_top_drawer --num_trajs 20
python rlds.py --env_name google_robot_open_middle_drawer --num_trajs 20
python rlds.py --env_name google_robot_open_bottom_drawer --num_trajs 20
```

### Closing drawers (different levels)
```bash
python rlds.py --env_name google_robot_close_top_drawer --num_trajs 20
python rlds.py --env_name google_robot_close_middle_drawer --num_trajs 20
python rlds.py --env_name google_robot_close_bottom_drawer --num_trajs 20
```

## **Placement Tasks**

```bash
python rlds.py --env_name google_robot_place_apple_in_closed_top_drawer --num_trajs 20
python rlds.py --env_name google_robot_place_in_closed_drawer --num_trajs 20
```

## **Moving Tasks** 

```bash
python rlds.py --env_name google_robot_move_near --num_trajs 20
```

---
## **Next Steps**

```bash
# Rebuild RLDS dataset with all collected episodes
cd simpler_env_switch_dataset
tfds build --overwrite

# Verify combined dataset
cd ..
python vis_data.py simpler_env_switch_dataset
```

## **Controls Reminder**

- **Left stick + L/ZL**: Translation (xyz)
- **Right stick + R/ZR**: Rotation (roll/pitch/yaw)
- **A/B buttons**: Gripper open/close
- **+ button**: Start episode / End episode